import re

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("miscovery/tokenizer")


# dataset
# Token length filter function
def is_within_limit(example, max_len=500):
    en_total = tokenizer(example["prompt_en"], example["response_en"], truncation=False, add_special_tokens=True)
    ar_total = tokenizer(example["prompt_ar"], example["response_ar"], truncation=False, add_special_tokens=True)
    return len(en_total["input_ids"]) <= max_len and len(ar_total["input_ids"]) <= max_len


# Function to replace \r\n with spaces
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'\r\n', ' ', text)
    return text


# Load datasets
ds = load_dataset("miscovery/Math_CoT_Arabic_English_Reasoning")["train"]

# Load and merge CSVs
csv1 = pd.read_csv("arithmatic_questions.csv")
csv2 = pd.read_csv("arithmatic.csv")
csv_combined = pd.concat([csv1, csv2]).reset_index(drop=True)

# Ensure required columns exist
required_cols = {"en_question", "ar_question", "en_answer", "ar_answer"}
assert required_cols.issubset(set(csv_combined.columns)), "CSV is missing required columns"

# Ensure answers are strings and clean \r\n
csv_combined["en_answer"] = csv_combined["en_answer"].astype(str).apply(clean_text)
csv_combined["ar_answer"] = csv_combined["ar_answer"].astype(str).apply(clean_text)
csv_dataset = Dataset.from_pandas(csv_combined)

# Convert base dataset answers to strings and clean \r\n
ds = ds.map(lambda x: {
    "en_answer": clean_text(str(x["en_answer"])),
    "ar_answer": clean_text(str(x["ar_answer"]))
}, batched=False)


# Generate prompt/response pairs
def to_prompt_response(example):
    return {
        "prompt_en": clean_text(f"{example['en_question']}"),
        "response_en": clean_text(example["en_answer"]),
        "prompt_ar": clean_text(f"{example['ar_question']}"),
        "response_ar": clean_text(example["ar_answer"])
    }


# Apply transformation
transformed_ds = ds.map(to_prompt_response)
transformed_csv = csv_dataset.map(to_prompt_response)

# Combine and filter by token length
combined = concatenate_datasets([transformed_ds, transformed_csv])
filtered_combined = combined.filter(is_within_limit)


# Flatten to prompt/response only
def flatten_bilingual_dataset(dataset):
    prompts = []
    for example in dataset:
        prompts.append({
            "prompt": example["prompt_en"],
            "response": example["response_en"]
        })
        prompts.append({
            "prompt": example["prompt_ar"],
            "response": example["response_ar"]
        })
    return Dataset.from_list(prompts)


final_dataset = flatten_bilingual_dataset(filtered_combined)

# Shuffle and split
final_dataset = final_dataset.shuffle(seed=42)
val_size = max(1, int(0.01 * len(final_dataset)))
val_dataset = final_dataset.select(range(val_size))
train_dataset = final_dataset.select(range(val_size, len(final_dataset)))

# Final structure
raw_datasets = {
    "train": train_dataset,
    "test": val_dataset
}


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


def load_checkpoint(checkpoint_path, model, device="cuda"):
    """Load model from checkpoint (simplified - only loads state dict)."""
    logger.info(f"Loading model from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

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
        model = load_checkpoint(args.checkpoint_path, model, device)

    # Create tensor datasets
    logger.info("Creating tensor datasets")
    cache_dir = args.cache_dir
    train_cache_file = f"{cache_dir}/train_tensors_{args.max_seq_length}.pt"
    test_cache_file = f"{cache_dir}/test_tensors_{args.max_seq_length}.pt"

    os.makedirs(cache_dir, exist_ok=True)

    dataset = create_tensor_datasets(
        raw_datasets['train']['prompt'],
        raw_datasets['train']['response'],
        tokenizer,
        max_length=args.max_seq_length,
        cache_file=train_cache_file
    )

    test_dataset = create_tensor_datasets(
        raw_datasets['test']['prompt'],
        raw_datasets['test']['response'],
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

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Calculate warmup steps as a ratio of total training steps
    if args.lr_warmup_ratio > 0:
        args.warmup_steps = int(total_steps * args.lr_warmup_ratio)
    else:
        args.warmup_steps = args.warmup_steps  # Use the existing value

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
        from transformers import get_polynomial_decay_schedule_with_warmup
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

    # Fine-tune the model
    logger.info("Starting fine-tuning")
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
        ema_decay=args.ema_decay
    )

    # Save final fine-tuned model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
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
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer", help="Tokenizer name or path")
    parser.add_argument("--checkpoint_path", type=str,
                        default="stage_01/output/model_final.pth",
                        help="Path to pre-trained checkpoint from Stage 1")
    parser.add_argument("--output_dir", type=str, default="stage_02/output",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="stage_01/cache",
                        help="Cache directory")

    # Add these to the argument parser in finetune.py
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "polynomial"],
                        help="Learning rate scheduler type")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="Factor to reduce LR by when using step scheduler")
    parser.add_argument("--lr_step_size", type=int, default=1000,
                        help="Number of steps between LR decreases in step scheduler")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum learning rate as a fraction of initial LR")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1,
                        help="Portion of training to use for warmup (as a ratio of total steps)")

    args, unknown = parser.parse_known_args()

    # Print arguments
    logger.info("Fine-tuning with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    main(args)
