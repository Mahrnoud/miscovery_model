"""
Enhanced training module with evaluation metrics and checkpoint management
"""
import math
import time

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


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


def train_model(model, train_dataloader, criterion, optimizer, scheduler,
                device, epochs=10, start_epoch=0, gradient_accumulation_steps=16,
                label_smoothing=0.1,
                weight_decay=0.01,
                max_grad_norm=1.0,
                ema_decay=0.9999,  # Added EMA
                eval_dataloader=None,  # Added evaluation dataloader
                tokenizer=None,  # Added tokenizer for evaluation
                eval_steps=500,  # Evaluate every N steps
                save_steps=1000,  # Save checkpoint every N steps
                output_dir="./outputs",  # Output directory
                max_checkpoints=2,  # Maximum number of checkpoints to keep
                ):
    """
    Enhanced training function with evaluation metrics and checkpoint management
    """
    # Import the evaluator class here to avoid circular imports
    from evaluation import ModelEvaluator

    model.train()
    global_step = 0
    best_eval_metric = float('-inf')

    # Create EMA for the model
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay)
    else:
        ema = None

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Initialize evaluator if evaluation is enabled
    evaluator = None
    if eval_dataloader is not None and tokenizer is not None:
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            eval_dataloader=eval_dataloader,
            device=device,
            output_dir=output_dir,
            max_checkpoints=max_checkpoints
        )

    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    autocast_device_type = 'cuda' if use_cuda else 'cpu'
    autocast_context = lambda: torch.amp.autocast(device_type=autocast_device_type, enabled=use_cuda)

    print(f"Starting training with the following parameters:")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Label smoothing: {label_smoothing}")
    print(f"- EMA decay: {ema_decay if ema is not None else 'Disabled'}")
    print(f"- Weight decay: {weight_decay}")
    print(f"- Max gradient norm: {max_grad_norm}")
    print(f"- Evaluation steps: {eval_steps if evaluator else 'Disabled'}")
    print(f"- Output directory: {output_dir}")

    # Track metrics for logging
    train_losses = []
    learning_rates = []
    eval_metrics = []

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

                # if global_step % 10 == 0:
                    # current_lr = scheduler.get_last_lr()[0]
                    # print(f"\nStep {global_step} - Current learning rate: {current_lr:.8f}")

                # Update EMA if enabled
                if ema is not None:
                    ema.update()

                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)
                train_losses.append(loss.item() * gradient_accumulation_steps)

                # Log info periodically
                if global_step % 100 == 0:
                    print(f"\nStep {global_step} - Current learning rate: {current_lr:.7f}")

                # Run evaluation at specified intervals
                if evaluator and global_step % eval_steps == 0:
                    # Apply EMA before evaluation if enabled
                    if ema is not None:
                        ema.apply_shadow()

                    # Get average training loss
                    avg_train_loss = sum(train_losses[-100:]) / min(100, len(train_losses[-100:]))

                    # Run evaluation
                    metrics = evaluator.evaluate(epoch + 1, global_step, train_loss=avg_train_loss)
                    eval_metrics.append(metrics)

                    # Restore model if EMA was applied
                    if ema is not None:
                        ema.restore()

                    # Reset model to training mode
                    model.train()

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

        # Run evaluation at the end of each epoch if not already done
        if evaluator and (global_step % eval_steps != 0):
            # Apply EMA before evaluation if enabled
            if ema is not None:
                ema.apply_shadow()

            # Get average training loss for the epoch
            avg_train_loss = avg_loss

            # Run evaluation
            metrics = evaluator.evaluate(epoch + 1, global_step, train_loss=avg_train_loss)
            eval_metrics.append(metrics)

            # Restore model if EMA was applied
            if ema is not None:
                ema.restore()

            # Reset model to training mode
            model.train()

    # Plot combined training history at the end
    if train_losses and evaluator:
        plot_training_history(train_losses, eval_metrics, output_dir)

    print("Training finished.")

    return model


def plot_training_history(train_losses, eval_metrics, output_dir):
    """
    Plot the complete training history
    """
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Create steps for x-axis
    train_steps = list(range(1, len(train_losses) + 1))
    eval_steps = [int(len(train_losses) * (i+1) / len(eval_metrics)) for i in range(len(eval_metrics))]

    # Extract evaluation metrics
    eval_losses = [m["eval_loss"] for m in eval_metrics]
    rouge_l = [m["rougeL"] for m in eval_metrics]
    bleu = [m["bleu"] for m in eval_metrics]
    f1 = [m["f1"] for m in eval_metrics]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # Plot losses
    ax1.plot(train_steps, train_losses, label="Training Loss", alpha=0.5, color="blue")
    ax1.plot(eval_steps, eval_losses, label="Evaluation Loss", marker="o", color="red")
    ax1.set_title("Training and Evaluation Loss")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot metrics
    ax2.plot(eval_steps, rouge_l, label="ROUGE-L", marker="o")
    ax2.plot(eval_steps, bleu, label="BLEU", marker="s")
    ax2.plot(eval_steps, f1, label="F1", marker="^")
    ax2.set_title("Evaluation Metrics")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "figures", "complete_training_history.png"), dpi=300)
    plt.close()
