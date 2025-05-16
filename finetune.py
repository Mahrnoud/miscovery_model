"""
Fine-tuning script for pre-trained Transformer models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
import random
import sys
import os
import argparse
import numpy as np

# Set up logging
import logging

# Import our modules
from model_architecture import Transformer
from enhanced_training_code import train_model
from evaluation import ModelEvaluator, evaluate_translations, load_best_model

# Import dataset preprocessing
from dataset_preprocessing import prepare_finetuning_dataset, get_tensor_datasets

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

    # Load and process datasets using the separate dataset preprocessing module
    logger.info("Preparing datasets for fine-tuning")
    raw_datasets = prepare_finetuning_dataset(tokenizer, args)

    # Convert to tensor datasets
    logger.info("Converting to tensor datasets")
    tensor_datasets = get_tensor_datasets(raw_datasets, tokenizer, args)

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_dataloader = DataLoader(
        tensor_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        tensor_datasets["test"],
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

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Calculate warmup steps as a ratio of total training steps
    args.warmup_steps = min(100, int(total_steps * 0.05))

    # Initialize the appropriate scheduler based on type
    if args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    elif args.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    elif args.lr_scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps
        )
    elif args.lr_scheduler_type == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            lr_end=args.learning_rate * args.min_lr_ratio,
            power=1.0
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
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
        eval_dataloader=test_dataloader,  # Added evaluation dataloader
        tokenizer=tokenizer,  # Added tokenizer for evaluation
        eval_steps=args.eval_steps,  # Evaluate every N steps
        save_steps=args.save_steps,  # Save checkpoint every N steps
        output_dir=args.output_dir,  # Output directory
        max_checkpoints=args.max_checkpoints  # Maximum number of checkpoints to keep
    )

    # Load best model for final evaluation
    model = load_best_model(model, args.output_dir, device)

    # Final evaluation on test set
    logger.info("Starting final evaluation on test set")
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
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=12, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # Evaluation parameters
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every N steps")
    parser.add_argument("--max_checkpoints", type=int, default=2, help="Maximum number of checkpoints to keep")

    # Learning rate scheduler parameters
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "polynomial"],
                        help="Learning rate scheduler type")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="Factor to reduce LR by when using step scheduler")
    parser.add_argument("--lr_step_size", type=int, default=1000,
                        help="Number of steps between LR decreases in step scheduler")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum learning rate as a fraction of initial LR")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.05,
                        help="Portion of training to use for warmup (as a ratio of total steps)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer", help="Tokenizer name or path")
    parser.add_argument("--checkpoint_path", type=str,
                        default="stage_01/output/best_model.pth",
                        help="Path to pre-trained checkpoint from Stage 1")
    parser.add_argument("--output_dir", type=str, default="stage_02/output",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="stage_02/cache",
                        help="Cache directory")
    parser.add_argument("--test_split", type=float, default=0.02, help="Test set split ratio")

    args, unknown = parser.parse_known_args()

    # Print arguments
    logger.info("Fine-tuning with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    main(args)
