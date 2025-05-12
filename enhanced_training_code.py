import math
import os
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

from enhanced_evaluation_code import save_checkpoint_optimized, evaluate_model_optimized

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
                early_stopping_patience=10,  # Increased from 5
                tokenizer=None,
                max_checkpoints=3,  # Increased from 2
                label_smoothing=0.1,
                weight_decay=0.01,
                max_grad_norm=1.0,
                ema_decay=0.9999  # Added EMA
                ):
    """
    Enhanced training function with additional features:
    - Label smoothing
    - Cosine learning rate schedule
    - Exponential Moving Average
    - Improved gradient clipping
    - Better logging
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

                # In enhanced_training_code.py, inside the train_model function
                # In the batch loop, after scheduler.step(), add:
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

                    eval_metrics = evaluate_model_optimized(
                        model,
                        test_dataloader,
                        criterion,
                        device,
                        tokenizer=tokenizer,
                        generate_samples=(global_step % (eval_steps * 2) == 0),
                        num_samples=2,
                        max_batches_per_eval=50,
                        seed=eval_seed,
                        label_smoothing=label_smoothing
                    )

                    # Restore original weights if EMA was applied
                    if ema is not None:
                        ema.restore()

                    avg_val_loss = eval_metrics['val_loss']
                    perplexity = eval_metrics['perplexity']
                    val_losses.append(avg_val_loss)

                    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Step {global_step} Eval:"
                          f" Val Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.4f}")

                    # Log metrics from evaluation if available
                    if 'rouge' in eval_metrics:
                        print(f"ROUGE scores: {eval_metrics['rouge']}")
                    if 'bleu' in eval_metrics:
                        print(f"BLEU score: {eval_metrics['bleu']:.4f}")

                    is_best = avg_val_loss < best_val_loss
                    # For best model checkpoint (when validation loss improves)
                    if is_best:
                        print(
                            f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model.")
                        best_val_loss = avg_val_loss
                        steps_since_last_improvement = 0

                        # Apply EMA if enabled
                        if ema is not None:
                            ema.apply_shadow()

                        # Use the optimized function with proper parameters
                        save_checkpoint_optimized(
                            model, optimizer, scheduler, epoch, loss.item() * gradient_accumulation_steps,
                            avg_val_loss, checkpoint_dir,
                            step=global_step, is_best=True,  # Set is_best to True
                            save_optimizer=True,  # Save optimizer state for best model
                            use_cpu_offload=False,
                            use_async=False,  # Don't use async for best model
                            metrics=eval_metrics  # Pass metrics
                        )

                        if ema is not None:
                            ema.restore()

                        print(f"Best model checkpoint saved at step {global_step}")
                    elif early_stopping_patience > 0:
                        steps_since_last_improvement += 1
                        print(f"Validation loss did not improve ({avg_val_loss:.4f} vs best {best_val_loss:.4f}). "
                              f"Patience: {steps_since_last_improvement}/{early_stopping_patience}")

                    if early_stopping_patience > 0 and steps_since_last_improvement >= early_stopping_patience:
                        print(
                            f"\nEarly stopping triggered after {early_stopping_patience} evaluation steps without improvement.")
                        early_stop_triggered = True
                        break

                    model.train()

                # For regular checkpoints
                if global_step % save_steps == 0 and not (global_step % eval_steps == 0 and is_best):
                    current_val_loss = avg_val_loss if global_step % eval_steps == 0 else None

                    save_checkpoint_optimized(
                        model, optimizer, scheduler, epoch, loss.item() * gradient_accumulation_steps,
                        current_val_loss, checkpoint_dir,
                        step=global_step, is_best=False,
                        save_optimizer=False,  # Don't need optimizer for regular checkpoints
                        use_cpu_offload=False,
                        use_async=True
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

        if early_stop_triggered:
            print("Exiting training loop due to early stopping.")
            break

    print("Training finished.")

    # Save the final model
    if ema is not None:
        ema.apply_shadow()

    save_checkpoint_optimized(
        model, optimizer, scheduler, epochs - 1, avg_loss,
        best_val_loss, checkpoint_dir,
        step=global_step, is_best=False, is_final=True,
        max_checkpoints=max_checkpoints,
        save_optimizer=True,  # Include optimizer for final model
        use_cpu_offload=False,
        use_async=False  # Don't use async for final important checkpoint
    )

    if ema is not None:
        ema.restore()

    return model
