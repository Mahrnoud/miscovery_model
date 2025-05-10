import os
import glob
import re
import threading
import copy
import torch
import random
from tqdm import tqdm
import time
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk


def evaluate_model_optimized(model, dataloader, criterion, device, tokenizer=None,
                             generate_samples=False, num_samples=3, seed=None,
                             label_smoothing=0.0, max_batches_per_eval=None):
    """
    Optimized evaluation function with reduced memory usage and faster processing

    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        tokenizer: Tokenizer for text generation
        generate_samples: Whether to generate text samples
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        label_smoothing: Label smoothing value for loss calculation
        max_batches_per_eval: Maximum number of batches to evaluate (None for all)

    Returns:
        Dictionary with evaluation metrics
    """

    # Record start time for performance monitoring
    start_time = time.time()

    model.eval()
    total_loss = 0
    total_batches = 0
    perplexities = []

    # For metrics calculation - only store a subset for memory efficiency
    all_predictions = []
    all_references = []
    max_examples_for_metrics = min(50, num_samples * 3)  # Limit examples for metrics

    # Use a different seed each time if none is provided
    if seed is None:
        seed = random.randint(0, 10000)

    # Set random seed for this evaluation run
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n------ Starting Optimized Evaluation (Seed: {seed}) ------")

    # Choose a fixed small set of batch indices for samples
    sample_batch_indices = []
    if tokenizer is not None and len(dataloader) > 0:
        # Calculate number of batches to evaluate
        num_batches = len(dataloader) if max_batches_per_eval is None else min(max_batches_per_eval, len(dataloader))

        # Choose at most 2 batches for samples
        if num_batches > 0:
            sample_batch_indices = random.sample(
                range(min(10, num_batches)),
                min(2, num_batches)
            )

    # Use autocast for mixed precision evaluation
    autocast = torch.amp.autocast('cuda')

    # Create a batch iterator with optional limiting
    limited_dataloader = dataloader
    if max_batches_per_eval is not None:
        limited_dataloader = [next(iter(dataloader)) for _ in range(min(max_batches_per_eval, len(dataloader)))]

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(tqdm(limited_dataloader, desc="Evaluating")):

            src, tgt = src.to(device), tgt.to(device)

            # Process samples for metrics - only from selected batches
            if tokenizer is not None and batch_idx in sample_batch_indices and len(
                    all_predictions) < max_examples_for_metrics:
                # Only choose 1-2 samples per batch for efficiency
                sample_indices = random.sample(
                    range(src.size(0)),
                    min(2, src.size(0))
                )

                for i in sample_indices:
                    # Skip if we already have enough examples
                    if len(all_predictions) >= max_examples_for_metrics:
                        break

                    # Decode reference text
                    response_text = tokenizer.decode(tgt[i], skip_special_tokens=True)
                    all_references.append(response_text)

                    # Generate prediction
                    with autocast:
                        generated = generate_text_optimized(
                            model,
                            src[i:i + 1],
                            tokenizer,
                            max_length=128,  # Shorter length for evaluation
                            device=device
                        )
                    all_predictions.append(generated)

                    # Clear individual tensors to save memory
                    del generated

            # Calculate loss
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with autocast:
                output = model(src, tgt_input)
                output_flat = output.contiguous().view(-1, output.size(-1))
                tgt_output_flat = tgt_output.contiguous().view(-1)

                if label_smoothing > 0:
                    loss = torch.nn.CrossEntropyLoss(
                        ignore_index=model.pad_idx if hasattr(model, 'pad_idx') else -100,
                        label_smoothing=label_smoothing
                    )(output_flat, tgt_output_flat)
                else:
                    loss = criterion(output_flat, tgt_output_flat)

            # Update metrics
            total_loss += loss.item()
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())
            total_batches += 1

    # Calculate metrics
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0

    metrics = {
        'val_loss': avg_loss,
        'perplexity': avg_perplexity,
        'eval_seed': seed,
        'eval_time': time.time() - start_time,
        'num_batches_evaluated': total_batches
    }

    print(f"\nEvaluation complete - Val Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")
    print(f"Evaluation time: {metrics['eval_time']:.2f} seconds")

    # Calculate ROUGE and BLEU only if we have predictions
    if all_predictions and all_references and len(all_predictions) == len(all_references):
        try:
            rouge = Rouge()
            rouge_scores = rouge.get_scores([p for p in all_predictions if p],
                                            [r for r in all_references if r],
                                            avg=True)
            metrics['rouge'] = {
                'rouge-1': rouge_scores['rouge-1']['f'],
                'rouge-2': rouge_scores['rouge-2']['f'],
                'rouge-l': rouge_scores['rouge-l']['f']
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")

        try:
            try:
                nltk.data.find('tokenizers/punkt')
            except:
                nltk.download('punkt')
                nltk.download('punkt_tab')

            # Tokenize predictions and references
            tokenized_preds = [nltk.word_tokenize(p.lower()) for p in all_predictions if p]
            tokenized_refs = [[nltk.word_tokenize(r.lower())] for r in all_references if r]

            if tokenized_preds and tokenized_refs and len(tokenized_preds) == len(tokenized_refs):
                smoothing = SmoothingFunction().method1
                bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
                metrics['bleu'] = bleu_score
        except Exception as e:
            print(f"Error calculating BLEU: {e}")

    # Only generate full samples if explicitly requested
    if generate_samples and tokenizer is not None:

        samples = []
        dataset = dataloader.dataset

        print("\nGenerating text samples...")

        # Use a different seed for generation
        gen_seed = random.randint(0, 10000)
        random.seed(gen_seed)
        torch.manual_seed(gen_seed)

        # Sample fewer indices for efficiency
        actual_samples = min(num_samples, 3)
        indices = random.sample(range(len(dataset)), min(actual_samples * 2, len(dataset)))

        for idx in indices[:actual_samples]:
            src, _ = dataset[idx]
            src = src.unsqueeze(0).to(device)

            sample_input = tokenizer.decode(src[0].tolist(), skip_special_tokens=True)

            # Use the optimized generation with reduced parameters
            generated = generate_text_optimized(
                model,
                src,
                tokenizer,
                max_length=128,
                device=device,
                temperature=0.8,
                top_p=0.92,
                do_sample=True
            )

            samples.append({
                'input': sample_input[:100] + ('...' if len(sample_input) > 100 else ''),  # Truncate for memory
                'generated': generated[:200] + ('...' if len(generated) > 200 else '')  # Truncate for memory
            })

            print(f"Sample {idx}: {generated[:100]}...")

        metrics['sample_count'] = len(samples)
        metrics['samples'] = samples
        metrics['generation_seed'] = gen_seed

    return metrics


def generate_text_optimized(model, src, tokenizer, max_length=128, device='cuda',
                            temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.2,
                            do_sample=True):
    """
    Memory-optimized text generation function with early stopping
    """
    model.eval()

    # Determine appropriate start token
    if tokenizer.bos_token_id is not None:
        start_token_id = tokenizer.bos_token_id
    elif tokenizer.cls_token_id is not None:
        start_token_id = tokenizer.cls_token_id
    elif tokenizer.pad_token_id is not None:
        start_token_id = tokenizer.pad_token_id
    else:
        start_token_id = 0

    # Early stopping tokens
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if tokenizer.sep_token_id is not None:
        stop_token_ids.append(tokenizer.sep_token_id)

    with torch.no_grad():
        decoder_input = torch.tensor([[start_token_id]], device=device)
        generated_ids = [start_token_id]

        # Use a fixed buffer size for repetition detection
        repetition_window = []
        max_window_size = 8

        # Use autocast for mixed precision
        with torch.amp.autocast('cuda', ):
            for _ in range(max_length):
                try:
                    # Create masks
                    src_mask, tgt_mask = None, None
                    if hasattr(model, 'create_masks'):
                        src_mask, tgt_mask = model.create_masks(src, decoder_input)

                    # Encoder output
                    if hasattr(model, 'encoder'):
                        enc_output = model.encoder(src, src_mask) if src_mask is not None else model.encoder(src)
                    else:
                        enc_output = None

                    # Decoder output
                    if hasattr(model, 'decoder') and enc_output is not None:
                        dec_output = model.decoder(decoder_input, enc_output, src_mask, tgt_mask)
                    else:
                        dec_output = model(src, decoder_input)

                    # Next token logits (only get the last position)
                    next_token_logits = dec_output[:, -1, :]

                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        for prev_token in set(generated_ids[-10:]):  # Only check recent tokens
                            next_token_logits[:, prev_token] /= repetition_penalty

                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                    # Apply nucleus sampling (top-p)
                    if top_p < 1.0 and do_sample:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[:, indices_to_remove] = float('-inf')

                    # Sample or greedy decoding
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

                    # Add to output
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                    next_token_id = next_token.item()
                    generated_ids.append(next_token_id)

                    # Update repetition tracker with fixed size
                    repetition_window.append(next_token_id)
                    if len(repetition_window) > max_window_size:
                        repetition_window.pop(0)

                    # Stop if end token
                    if next_token_id in stop_token_ids:
                        break

                    # Stop for repetition (same token repeated)
                    if len(repetition_window) >= 4 and len(set(repetition_window[-4:])) == 1:
                        break

                    # Stop for pattern repetition
                    if len(repetition_window) >= 6:
                        half = len(repetition_window) // 2
                        if repetition_window[-half:] == repetition_window[-2 * half:-half]:
                            break

                except Exception as e:
                    print(f"Error in generation: {e}")
                    break

        # Decode text
        try:
            generated_text = tokenizer.decode(decoder_input[0].tolist(), skip_special_tokens=True)
            return generated_text if generated_text.strip() else "[Empty generation]"
        except Exception as e:
            return f"[Error: {str(e)}]"


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

    # Force garbage collection
    checkpoint = None
