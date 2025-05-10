import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import random
import sys
from tqdm import tqdm
import os
import argparse
import numpy as np

# Standard library imports
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
import logging

from model_architecture import Transformer
from enhanced_training_code import train_model


def create_tensor_datasets(prompts, responses, tokenizer, max_length=512, cache_file=None):
    # First check if we should load from cache
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading preprocessed tensors from {cache_file}")
        try:
            return torch.load(cache_file)
        except Exception as e:
            logger.warning(f"Error loading cache file: {e}")
            logger.info("Falling back to processing data from scratch")

    logger.info("Processing data into tensors...")
    input_tensors = []
    target_tensors = []

    for p, r in tqdm(zip(prompts, responses), total=len(prompts), desc="Creating tensors"):
        input_encoding = tokenizer(
            p,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        target_encoding = tokenizer(
            r,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_tensors.append(input_encoding["input_ids"].squeeze(0))
        target_tensors.append(target_encoding["input_ids"].squeeze(0))

    input_tensors = torch.stack(input_tensors)
    target_tensors = torch.stack(target_tensors)

    dataset = TensorDataset(input_tensors, target_tensors)

    if cache_file:
        logger.info(f"Saving preprocessed tensors to {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        try:
            torch.save(dataset, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
            logger.info("Continuing without caching")

    return dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Enable deterministic behavior for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cuda"):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer and scheduler if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return additional training info
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)

    metrics = {}
    for key in checkpoint:
        if key.startswith("metric_"):
            metrics[key[7:]] = checkpoint[key]

    logger.info(f"Loaded checkpoint from epoch {epoch}, step {step}")
    if metrics:
        logger.info(f"Metrics from checkpoint: {metrics}")

    # Print current learning rate
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    return model, optimizer, scheduler, epoch, step


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create model instance
    logger.info("Initializing model architecture")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        vocab_size=len(tokenizer),
        max_len=args.max_seq_length,
        pad_idx=tokenizer.pad_token_id,
        dropout=args.dropout
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay
    )

    # Initialize learning rate scheduler (will be updated after loading checkpoint)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=100000,
        min_lr_ratio=0.1
    )

    # Load pre-trained model from Stage 1
    start_epoch = 0
    if args.checkpoint_path:
        model, optimizer, scheduler, loaded_epoch, step = load_checkpoint(
            args.checkpoint_path, model, optimizer, scheduler, device
        )
        start_epoch = loaded_epoch + 1  # Start from the next epoch

    # Create tensor datasets
    logger.info("Creating tensor datasets")
    cache_dir = args.cache_dir
    train_cache_file = f"{cache_dir}/train_tensors_{args.max_seq_length}.pt"
    test_cache_file = f"{cache_dir}/test_tensors_{args.max_seq_length}.pt"

    os.makedirs(cache_dir, exist_ok=True)

    dataset = create_tensor_datasets(
        # raw_datasets['train']['prompt'],
        # raw_datasets['train']['response'],
        None,
        None,
        tokenizer,
        max_length=args.max_seq_length,
        cache_file=train_cache_file
    )

    test_dataset = create_tensor_datasets(
        # raw_datasets['test']['prompt'],
        # raw_datasets['test']['response'],
        None,
        None,
        tokenizer,
        max_length=args.max_seq_length,
        cache_file=test_cache_file
    )

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Check first few batches
    for i, (src, tgt) in enumerate(train_dataloader):
        logger.info(f"Batch {i + 1}: Source shape: {src.shape}, Target shape: {tgt.shape}")
        if i >= 2:
            break

    # Update scheduler with actual training steps
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )

    # Initialize loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=args.label_smoothing
    )

    # Log info about the model and training
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Training examples: {len(train_dataloader)}")
    logger.info(f"Evaluation examples: {len(test_dataloader)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Total training steps: {total_steps}")

    # Fine-tune the model
    logger.info(f"Starting fine-tuning from epoch {start_epoch}")

    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.num_epochs,
        start_epoch=start_epoch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
        tokenizer=tokenizer,
        max_checkpoints=args.max_checkpoints,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        ema_decay=args.ema_decay
    )

    # Save final fine-tuned model
    final_model_path = os.path.join(args.output_dir, f"final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final fine-tuned model to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model on downstream tasks")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=12, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=26, help="Number of epochs")
    parser.add_argument("--eval_steps", type=int, default=5600, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=40000, help="Save steps")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_checkpoints", type=int, default=2, help="Maximum number of checkpoints to keep")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer", help="Tokenizer name or path")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/content/drive/MyDrive/miscovery/stage_02_v2/output_2/checkpoints/best_model.pth",
                        help="Path to pre-trained checkpoint from Stage 1")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/miscovery/stage_02_v2/output_3",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="/content/drive/MyDrive/miscovery/stage_02_v2/cache_1",
                        help="Cache directory")

    args, unknown = parser.parse_known_args()

    # Print arguments
    logger.info("Fine-tuning with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    main(args)
