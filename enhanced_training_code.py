import math
import os
import random
import time
import gc

import torch
import torch.nn as nn
from tqdm import tqdm

# Import enhanced evaluation
from enhanced_evaluation import (
    EvaluationCallback,
    greedy_decode,
    sample_decode,
    plot_training_history
)

import rouge
import nltk

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')


# Define metric calculation functions within this file for ease of use

def calculate_rouge(predictions, references):
    """Calculate ROUGE scores for a list of predictions and references"""
    rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)

    # Ensure predictions and references are not empty
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    preds, refs = zip(*valid_pairs)

    try:
        scores = rouge_evaluator.get_scores(preds, refs, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def calculate_bleu(predictions, references):
    """Calculate BLEU score for a list of predictions and references"""
    # Tokenize predictions and references
    tokenized_predictions = [nltk.word_tokenize(p.lower()) for p in predictions]
    tokenized_references = [[nltk.word_tokenize(r.lower())] for r in references]

    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1
    try:
        score = corpus_bleu(tokenized_references, tokenized_predictions, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0.0


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0,
                                    last_epoch=-1):
    """
    Create a schedule with a cosine learning rate decay and linear warmup.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Adjust for minimum learning rate
        decay_factor = decay_factor * (1.0 - min_lr_ratio) + min_lr_ratio

        return decay_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def save_checkpoint_optimized(model, optimizer, scheduler, epoch, loss, val_loss, checkpoint_dir,
                              step=None, is_best=False, is_final=False, max_checkpoints=3, metrics=None,
                              save_optimizer=True, use_cpu_offload=False, use_async=False):
    """
    Memory-efficient checkpoint saving with CPU offloading and async options

    Args:
        model: Model to save
        optimizer: Optimizer to save (optional if save_optimizer=False)
        scheduler: Scheduler to save (optional if save_optimizer=False)
        epoch: Current epoch number
        loss: Current loss value
        val_loss: Validation loss value
        checkpoint_dir: Directory to save checkpoints
        step: Current step number (optional)
        is_best: Whether this is the best model so far
        is_final: Whether this is the final model
        max_checkpoints: Maximum number of checkpoints to keep
        metrics: Additional metrics to save (optional)
        save_optimizer: Whether to save optimizer and scheduler (False for smaller checkpoints)
        use_cpu_offload: Whether to offload tensors to CPU before saving
        use_async: Whether to save checkpoint asynchronously (experimental)
    """
    import os
    import glob
    import re
    import threading
    import copy

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a minimal or full checkpoint depending on configuration
    if save_optimizer:
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': loss,
            'val_loss': val_loss
        }
    else:
        # Model-only checkpoint (smaller size)
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'train_loss': loss,
            'val_loss': val_loss
        }

    # Add metrics if provided (exclude large sample data)
    if metrics is not None:
        for key, value in metrics.items():
            if key not in checkpoint and key != 'samples':
                # Convert NumPy values to Python native types for JSON serialization
                if hasattr(value, 'item'):
                    value = value.item()  # Convert PyTorch tensor to Python scalar
                elif hasattr(value, 'tolist'):
                    value = value.tolist()  # Convert NumPy array to list
                checkpoint[f'metric_{key}'] = value

    # CPU offloading for reduced GPU memory pressure
    if use_cpu_offload and torch.cuda.is_available():
        # Create a new state dict with CPU tensors
        original_state_dict = checkpoint['model_state_dict']
        cpu_state_dict = {k: v.cpu() for k, v in original_state_dict.items()}
        checkpoint['model_state_dict'] = cpu_state_dict

        # Also move optimizer states to CPU if present
        if save_optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            # Handle the optimizer state dict structure
            optimizer_dict = checkpoint['optimizer_state_dict']
            for param_group in optimizer_dict['param_groups']:
                for param_id in param_group['params']:
                    if param_id in optimizer_dict['state']:
                        for k, v in optimizer_dict['state'][param_id].items():
                            if torch.is_tensor(v):
                                optimizer_dict['state'][param_id][k] = v.cpu()

    # Define save function for either direct or async saving
    def _save_checkpoint(checkpoint, path):
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    # Define cleanup function for old checkpoints
    def _cleanup_old_checkpoints(checkpoint_pattern, max_to_keep):
        checkpoint_files = glob.glob(checkpoint_pattern)
        if len(checkpoint_files) <= max_to_keep:
            return

        # Extract numbers and sort
        numbered_files = []
        for file_path in checkpoint_files:
            match = re.search(r'(\d+)\.pth', file_path)
            if match:
                numbered_files.append((int(match.group(1)), file_path))

        # Keep newest, delete oldest
        numbered_files.sort(reverse=True)
        for _, file_path in numbered_files[max_to_keep:]:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                print(f"Error removing checkpoint {file_path}: {e}")

    # Save checkpoints based on type
    paths_to_save = []

    # Final model
    if is_final:
        paths_to_save.append(f"{checkpoint_dir}/model_final.pth")

    # Best model
    if is_best:
        paths_to_save.append(f"{checkpoint_dir}/best_model.pth")

    # Step or epoch checkpoint
    if step is not None:
        paths_to_save.append(f"{checkpoint_dir}/checkpoint_step_{step}.pth")
        cleanup_pattern = f"{checkpoint_dir}/checkpoint_step_*.pth"
    else:
        paths_to_save.append(f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth")
        cleanup_pattern = f"{checkpoint_dir}/checkpoint_epoch_*.pth"

    # Save the checkpoint(s)
    if use_async and not is_best and not is_final:
        # Only use async for regular checkpoints, not best or final
        # Copy the checkpoint to avoid race conditions
        checkpoint_copy = copy.deepcopy(checkpoint)
        for path in paths_to_save:
            threading.Thread(
                target=_save_checkpoint,
                args=(checkpoint_copy, path)
            ).start()

        # Clean up in background too
        threading.Thread(
            target=_cleanup_old_checkpoints,
            args=(cleanup_pattern, max_checkpoints)
        ).start()
    else:
        # Synchronous saving for important checkpoints
        for path in paths_to_save:
            _save_checkpoint(checkpoint, path)

        # Clean up old checkpoints
        _cleanup_old_checkpoints(cleanup_pattern, max_checkpoints)


class EMA:
    """
    Exponential Moving Average for model parameters
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler,
                device, epochs=10, start_epoch=0, gradient_accumulation_steps=16,
                checkpoint_dir="models/checkpoints",
                eval_steps=3500,
                save_steps=5000,
                early_stopping_patience=10,
                tokenizer=None,
                max_checkpoints=3,
                label_smoothing=0.1,
                weight_decay=0.01,
                max_grad_norm=1.0,
                ema_decay=0.9999
                ):
    """
    Enhanced training function with additional features:
    - Label smoothing
    - Cosine learning rate schedule
    - Exponential Moving Average
    - Improved gradient clipping
    - Better logging
    - Enhanced evaluation
    - Memory leak fixes
    """
    model.train()
    best_val_loss = float('inf')
    global_step = 0
    steps_since_last_improvement = 0
    early_stop_triggered = False
    last_eval_seed = None

    avg_loss = None  # Initialize at the start

    # Create EMA for the model
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay)
    else:
        ema = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    autocast_device_type = 'cuda' if use_cuda else 'cpu'
    autocast_context = lambda: torch.amp.autocast(device_type=autocast_device_type, enabled=use_cuda)

    print(f"Starting training with the following parameters:")
    print(f"- Early stopping patience: {early_stopping_patience if early_stopping_patience > 0 else 'Disabled'}")
    print(f"- Max checkpoints: {max_checkpoints}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Label smoothing: {label_smoothing}")
    print(f"- EMA decay: {ema_decay if ema is not None else 'Disabled'}")
    print(f"- Weight decay: {weight_decay}")
    print(f"- Max gradient norm: {max_grad_norm}")

    # Track metrics for logging
    train_losses = []
    val_losses = []
    learning_rates = []

    # Initialize the evaluation callback
    eval_callback = EvaluationCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=test_dataloader,
        device=device,
        output_dir=os.path.join(checkpoint_dir, "eval_results"),
        eval_steps=eval_steps,
        save_best_model=True,
        handle_arabic=True,
        max_eval_samples=3
    )

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        optimizer.zero_grad()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        epoch_start_time = time.time()

        for batch_idx, (src, tgt) in enumerate(batch_iterator):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with autocast_context():
                try:
                    output = model(src, tgt_input)
                    output_flat = output.contiguous().view(-1, output.size(-1))
                    tgt_output_flat = tgt_output.contiguous().view(-1)

                    # Apply label smoothing if specified
                    if label_smoothing > 0:
                        loss = nn.CrossEntropyLoss(
                            ignore_index=model.pad_idx,
                            label_smoothing=label_smoothing
                        )(output_flat, tgt_output_flat)
                    else:
                        loss = criterion(output_flat, tgt_output_flat)

                    loss = loss / gradient_accumulation_steps
                except RuntimeError as e:
                    print(f"\nError during forward pass: {e}")
                    print(f"Source shape: {src.shape}, Target input shape: {tgt_input.shape}")
                    raise e

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)

                # Improved gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Add this line to track metrics
                eval_callback.on_step_end(global_step, loss.item() * gradient_accumulation_steps, optimizer)

                # Log learning rate periodically
                if global_step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"\nStep {global_step} - Current learning rate: {current_lr:.7f}")

                # Update EMA if enabled
                if ema is not None:
                    ema.update()

                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)
                train_losses.append(loss.item() * gradient_accumulation_steps)

                # Evaluation at specified steps
                if global_step % eval_steps == 0:
                    # Generate a new seed for each evaluation
                    eval_seed = random.randint(0, 10000)
                    # Ensure we don't use the same seed as last time
                    while eval_seed == last_eval_seed and last_eval_seed is not None:
                        eval_seed = random.randint(0, 10000)

                    last_eval_seed = eval_seed

                    # Apply EMA weights for evaluation if enabled
                    if ema is not None:
                        ema.apply_shadow()

                    # Explicitly set model to eval mode
                    model.eval()

                    # Use evaluation from eval_callback
                    eval_callback.on_evaluation(global_step)

                    # Get evaluation metrics from the callback
                    avg_val_loss = 0.0  # Default value if not available
                    if eval_callback.val_losses and len(eval_callback.val_losses) > 0:
                        avg_val_loss = eval_callback.val_losses[-1]

                    # Check for best model
                    is_best = False
                    if avg_val_loss < best_val_loss:
                        print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f})")
                        best_val_loss = avg_val_loss
                        steps_since_last_improvement = 0
                        is_best = True
                    else:
                        steps_since_last_improvement += 1
                        print(f"Validation loss did not improve ({avg_val_loss:.4f} vs best {best_val_loss:.4f}). "
                              f"Patience: {steps_since_last_improvement}/{early_stopping_patience}")

                    # Restore original weights if EMA was applied
                    if ema is not None:
                        ema.restore()

                    # Save checkpoint
                    save_checkpoint_optimized(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=loss.item() * gradient_accumulation_steps,
                        val_loss=avg_val_loss,
                        checkpoint_dir=checkpoint_dir,
                        step=global_step,
                        is_best=is_best,
                        save_optimizer=True,
                        max_checkpoints=max_checkpoints
                    )

                    # Check early stopping
                    if early_stopping_patience > 0 and steps_since_last_improvement >= early_stopping_patience:
                        print(
                            f"\nEarly stopping triggered after {early_stopping_patience} evaluation steps without improvement.")
                        early_stop_triggered = True
                        break

                    # Set model back to train mode
                    model.train()

                    # Clear CUDA cache after evaluation to prevent memory leaks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Force garbage collection
                    gc.collect()

                # Regular checkpoint saving at specified steps
                elif global_step % save_steps == 0:
                    save_checkpoint_optimized(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=loss.item() * gradient_accumulation_steps,
                        val_loss=None,  # No validation loss for regular checkpoint
                        checkpoint_dir=checkpoint_dir,
                        step=global_step,
                        is_best=False,
                        save_optimizer=False,
                        use_async=True,
                        max_checkpoints=max_checkpoints
                    )
                    print(f"\nRegular checkpoint saved at step {global_step}")

            total_loss += loss.item() * gradient_accumulation_steps

            batch_iterator.set_postfix({
                'Loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.7f}",
                'Step': global_step
            })

        batch_iterator.close()
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. Average Train Loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        save_checkpoint_optimized(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=avg_loss,
            val_loss=None,  # No validation loss for epoch checkpoint
            checkpoint_dir=checkpoint_dir,
            step=global_step,
            is_best=False,
            is_final=False,
            save_optimizer=True,
            max_checkpoints=max_checkpoints
        )
        print(f"Epoch {epoch + 1} checkpoint saved")

        # Clear CUDA cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if early_stop_triggered:
            print("Exiting training loop due to early stopping.")
            break

    # At the end of training
    eval_callback.on_training_end()

    print("Training finished.")

    # Save the final model
    if ema is not None:
        ema.apply_shadow()

    save_checkpoint_optimized(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epochs - 1,
        loss=avg_loss,
        val_loss=best_val_loss,
        checkpoint_dir=checkpoint_dir,
        step=global_step,
        is_best=False,
        is_final=True,
        max_checkpoints=max_checkpoints,
        save_optimizer=True,
        use_async=False  # Don't use async for final important checkpoint
    )
    print(f"Final model saved to {checkpoint_dir}/model_final.pth")

    if ema is not None:
        ema.restore()

    # Create final training vs validation plot
    plot_training_history(
        train_losses=eval_callback.train_losses,
        val_losses=eval_callback.val_losses,
        lr_history=eval_callback.lr_history,
        output_dir=checkpoint_dir
    )

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model
