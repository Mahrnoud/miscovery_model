"""
Enhanced Transformer-based Model for Text Summarization and Translation
Main training script with improved architecture, training process, and evaluation
"""
import os
import random
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
# Standard library imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import our model and training utilities
from model_architecture import Transformer
from enhanced_training_code import train_model
from evaluation import ModelEvaluator, evaluate_translations, load_best_model

# Import dataset preprocessing
from dataset_preprocessing import prepare_training_dataset, get_tensor_datasets

# Set tokenizers parallelism to False to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
import logging

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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1_000_000_000:
        logger.info(f"Model Parameters: {total / 1_000_000_000:.2f}B")
    elif total >= 1_000_000:
        logger.info(f"Model Parameters: {total / 1_000_000:.2f}M")
    else:
        logger.info(f"Model Parameters: {total:,}")

    return total


def main(args):
    # Start time
    print(f"Starting script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Load and process datasets using the separate dataset preprocessing module
    logger.info("Preparing datasets")
    raw_datasets = prepare_training_dataset(tokenizer, args)

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

    # Initialize model
    logger.info("Initializing model")
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

    # Count model parameters
    count_parameters(model)

    # Initialize loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=args.label_smoothing
    )

    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay
    )

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Initialize learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        # min_lr_ratio=0.1  # Minimum learning rate will be 10% of max
    )

    # Create directory structure
    logger.info("Setting up directories")
    saving_directory = args.output_dir
    os.makedirs(saving_directory, exist_ok=True)

    # Train the model with evaluation
    logger.info("Starting training with evaluation")
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
        output_dir=saving_directory,  # Output directory
        max_checkpoints=args.max_checkpoints  # Maximum number of checkpoints to keep
    )

    # Load best model for final evaluation
    model = load_best_model(model, saving_directory, device)

    # Final evaluation on test set
    logger.info("Starting final evaluation on test set")
    evaluator = ModelEvaluator(model, tokenizer, test_dataloader, device, saving_directory)
    final_metrics = evaluator.evaluate(args.num_epochs, total_steps)

    # Show translation examples
    evaluate_translations(model, test_dataloader, tokenizer, device, num_examples=5)

    # Save final model
    logger.info(f"Saving final model to {saving_directory}/model_final.pth")
    torch.save(model.state_dict(), f"{saving_directory}/model_final.pth")
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enhanced Transformer model")

    # Model architecture parameters
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension (4x d_model)")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=12, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    # Evaluation parameters (new)
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every N steps")
    parser.add_argument("--max_checkpoints", type=int, default=2, help="Maximum number of checkpoints to keep")

    # Dataset parameters
    parser.add_argument("--samples_per_lang", type=int, default=1000, help="Number of samples per language")
    parser.add_argument("--test_split", type=float, default=0.02, help="Test set split ratio")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer", help="Tokenizer name or path")
    parser.add_argument("--output_dir", type=str, default="stage_01/output",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="stage_01/cache",
                        help="Cache directory")

    args, unknown = parser.parse_known_args()

    # Print arguments
    logger.info("Training with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    main(args)
