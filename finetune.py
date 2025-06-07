"""
Fine-tuning script for pre-trained Transformer models
Updated to support separate test directories
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)
import random
import sys
import os
import argparse
import numpy as np

from custom_lr_scheduler import create_custom_scheduler

# Set up logging
import logging

# Import our modules
from model_architecture import Transformer
from enhanced_training_code import train_model
from evaluation import ModelEvaluator, evaluate_translations, load_best_model

# Import dataset preprocessing
from dataset_preprocessing import get_tensor_datasets_memory_efficient, prepare_high_quality_training_dataset_memory_efficient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetuning.log"),
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


def initialize_scheduler_finetune(optimizer, args, total_steps):
    """
    Initialize learning rate scheduler for fine-tuning
    """
    print(f"Initializing {args.lr_scheduler_type} scheduler for fine-tuning with:")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Warmup steps: {args.warmup_steps}")
    print(f"  - Initial LR: {args.learning_rate}")

    if hasattr(args, 'min_lr_ratio'):
        print(f"  - Min LR ratio: {args.min_lr_ratio}")
        print(f"  - Min LR: {args.learning_rate * args.min_lr_ratio}")
    if hasattr(args, 'decay_start_step'):
        print(f"  - Decay start step: {args.decay_start_step}")

    return create_custom_scheduler(optimizer, args, total_steps)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    """Load model and optimizer state from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Set weights_only=False explicitly to handle the error
    try:
        # First try with weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info("Successfully loaded checkpoint with weights_only=False")
    except Exception as e:
        logger.warning(f"Error loading with weights_only=False: {e}")
        # If that fails, try with weights_only=True
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            logger.info("Successfully loaded checkpoint with weights_only=True")
        except Exception as e2:
            logger.error(f"Failed to load checkpoint with both options: {e2}")
            raise e2

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")

    logger.info(f"Model loaded successfully")
    return model


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

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

    # Load pre-trained model from Stage 1 if specified
    if args.checkpoint_path:
        model = load_checkpoint(args.checkpoint_path, model, optimizer, device)

    # Load and process datasets using the BALANCED approach (recommended)
    logger.info("Preparing datasets for fine-tuning")
    raw_datasets = prepare_high_quality_training_dataset_memory_efficient(
        train_csv_directory=args.train_data_dir,
        test_csv_directory=args.test_data_dir,
        test_split_ratio=args.test_split,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        processing_method="balanced",  # NEW: Balanced approach
        chunk_size=10000,  # NEW: Larger chunks = faster processing
        cache_chunks=10  # NEW: Keep 5 chunks in memory = faster training
    )

    # Skip tensor conversion (already done)
    tensor_datasets = get_tensor_datasets_memory_efficient(raw_datasets, tokenizer, args)

    # Create dataloaders with optimized settings
    train_dataloader = DataLoader(
        tensor_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # Can use workers now
        pin_memory=True,
        prefetch_factor=2  # Prefetch for speed
    )

    test_dataloader = DataLoader(
        tensor_datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    # Check first few batches
    for i, (src, tgt) in enumerate(train_dataloader):
        logger.info(f"Batch {i + 1}: Source shape: {src.shape}, Target shape: {tgt.shape}")
        if i >= 2:
            break

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Calculate warmup steps as a ratio of total training steps
    args.warmup_steps = min(100, int(total_steps * 0.05))

    # Initialize the appropriate scheduler based on type
    scheduler = initialize_scheduler_finetune(optimizer, args, total_steps)

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

    # Fine-tune the model with evaluation
    logger.info("Starting fine-tuning with evaluation")
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        ema_decay=args.ema_decay,
        eval_dataloader=test_dataloader,  # This now uses the entire test dataset
        tokenizer=tokenizer,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        max_checkpoints=args.max_checkpoints
    )

    # Load best model for final evaluation
    model = load_best_model(model, args.output_dir, device)

    # Final evaluation on test set (entire test dataset)
    logger.info("Starting final evaluation on entire test set")
    evaluator = ModelEvaluator(model, tokenizer, test_dataloader, device, args.output_dir)
    final_metrics = evaluator.evaluate(args.num_epochs, total_steps)

    # Show translation examples
    evaluate_translations(model, test_dataloader, tokenizer, device, num_examples=5)

    # Save final fine-tuned model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": final_metrics
        },
        final_model_path
    )
    logger.info(f"Saved final fine-tuned model to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model on downstream tasks")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    # Evaluation parameters
    parser.add_argument("--eval_steps", type=int, default=10000, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every N steps")
    parser.add_argument("--max_checkpoints", type=int, default=12, help="Maximum number of checkpoints to keep")

    # Learning rate scheduler parameters
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        choices=["constant", "linear_decay_to_min", "cosine_decay_to_min",
                                 "step_decay_to_min", "cosine", "linear"],
                        help="Learning rate scheduler type")

    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum learning rate as ratio of initial LR (for decay schedulers)")

    parser.add_argument("--decay_start_step", type=int, default=0,
                        help="Step to start LR decay (0 = start immediately after warmup)")

    parser.add_argument("--lr_decay_factor", type=float, default=0.5,
                        help="Factor to multiply LR by in step decay scheduler")

    parser.add_argument("--lr_step_size", type=int, default=1000,
                        help="Number of steps between LR decreases in step scheduler")

    # Dataset parameters - UPDATED
    parser.add_argument("--train_data_dir", type=str, default="/kaggle/working/1_Dataset_May_2025/Train",
                        help="Directory containing training CSV files")
    parser.add_argument("--test_data_dir", type=str, default="/kaggle/working/1_Dataset_May_2025/Test",
                        help="Directory containing test CSV files (optional)")
    parser.add_argument("--test_split", type=float, default=0.02,
                        help="Test set split ratio (only used if test_data_dir is None)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer_v2", help="Tokenizer name or path")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/kaggle/working/checkpoint.pth",
                        help="Path to pre-trained checkpoint from Stage 1")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/256_v2/stage_01_3/output",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="/kaggle/working/256_v2/stage_01_3/cache",
                        help="Cache directory")

    args, unknown = parser.parse_known_args()

    # Print arguments
    logger.info("Fine-tuning with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    main(args)
