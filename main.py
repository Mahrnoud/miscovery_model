"""
Enhanced Transformer-based Model for Text Summarization and Translation
Main training script with improved architecture and training process
(Modified: Removed evaluation and checkpoint functionality)
"""
import os
import random

import sys
import argparse
from datetime import datetime

import numpy as np
# Standard library imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

# Import our model and training utilities
from model_architecture import Transformer
from enhanced_training_code import train_model, get_cosine_schedule_with_warmup

# Dataset preprocessing imports
import nltk
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from langdetect import detect, LangDetectException

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


# Re-using your original dataset preprocessing functions
# Note: This section would include your original preprocessing code
# that you asked to keep the same
MAX_SEQ_LENGTH = 128
MIN_WORDS_FOR_SHUFFLE = 4


# Helper functions for text processing
def is_english(text):
    try:
        sample_text = text[:500].replace("[LANG_EN]", "").replace("[LANG_AR]", "").strip()
        if not sample_text:
            return any(ord(char) < 128 for char in text)
        return detect(sample_text) == 'en'
    except LangDetectException:
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        total_chars = len(text)
        if total_chars == 0: return True
        return (ascii_chars / total_chars) > 0.5


def shuffle_words_in_text(text):
    words = text.split()
    if len(words) < MIN_WORDS_FOR_SHUFFLE:
        return text, False
    shuffled_words = words[:]
    random.shuffle(shuffled_words)
    if shuffled_words == words and len(words) > 1:
        idx1, idx2 = random.sample(range(len(words)), 2)
        shuffled_words[idx1], shuffled_words[idx2] = shuffled_words[idx2], shuffled_words[idx1]
    return " ".join(shuffled_words), True


# Core preprocessing function for creating sentence pairs
def create_shuffled_sentence_pairs(examples, tokenizer,
                                   max_length=MAX_SEQ_LENGTH,
                                   add_special_tokens=True):
    prompts = []
    responses = []
    skipped_sentences = 0
    processed_articles = 0

    text_key = 'article' if 'article' in examples else 'text'
    if text_key not in examples:
        raise KeyError(f"Could not find 'article' or 'text' column in examples: {list(examples.keys())}")

    article_iterator = examples[text_key]

    for article_text in article_iterator:
        processed_articles += 1
        if article_text is None or not isinstance(article_text, str) or not article_text.strip():
            continue

        lang_tag = "[LANG_EN]" if is_english(article_text) else "[LANG_AR]"

        try:
            sentences = nltk.sent_tokenize(article_text)
        except Exception as e:
            continue

        for original_sentence in sentences:
            original_sentence = original_sentence.strip()
            if not original_sentence:
                continue

            shuffled_sentence, was_shuffled = shuffle_words_in_text(original_sentence)
            if not was_shuffled:
                skipped_sentences += 1
                continue

            prompt_core = shuffled_sentence
            response_core = original_sentence

            if add_special_tokens:
                prompt_str = f"{lang_tag} {prompt_core}"
                response_str = f"{lang_tag} {response_core}"
            else:
                prompt_str = prompt_core
                response_str = response_core

            prompt_tokens = tokenizer(prompt_str, max_length=max_length, truncation=True, padding=False)["input_ids"]
            response_tokens = tokenizer(response_str, max_length=max_length, truncation=True, padding=False)[
                "input_ids"]

            final_prompt_str = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            final_response_str = tokenizer.decode(response_tokens, skip_special_tokens=False)

            prompts.append(final_prompt_str)
            responses.append(final_response_str)

    return {"prompt": prompts, "response": responses}


def create_tensor_datasets(prompts, responses, tokenizer, max_length=512, cache_file=None):
    # First check if we should load from cache
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading preprocessed tensors from {cache_file}")
        try:
            return torch.load(cache_file, weights_only=False)
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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1_000_000_000:
        logger.info(f"Model Parameters: {total / 1_000_000_000:.2f}B")
    elif total >= 1_000_000:
        logger.info(f"Model Parameters: {total / 1_000_000:.2f}M")
    else:
        logger.info(f"Model Parameters: {total:,}")

    print(f"Model Parameters: {total}")
    return total


def main(args):
    # Start time
    print(f"Starting script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Load and process datasets
    logger.info("Loading datasets")
    en_dataset_stream = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    ar_dataset_stream = load_dataset("csebuetnlp/xlsum", "arabic", split="train", streaming=True)

    # Sample from datasets
    NUM_SAMPLES_PER_LANG = args.samples_per_lang
    logger.info(f"Sampling {NUM_SAMPLES_PER_LANG} examples from each language dataset")

    en_samples = list(en_dataset_stream.take(NUM_SAMPLES_PER_LANG))
    ar_samples = list(ar_dataset_stream.take(NUM_SAMPLES_PER_LANG))

    en_temp_dataset = Dataset.from_dict(
        {'article': [sample['article'] for sample in en_samples if sample.get('article')]})
    ar_temp_dataset = Dataset.from_dict({'text': [sample['text'] for sample in ar_samples if sample.get('text')]})

    # Process datasets
    logger.info("Processing English dataset")
    processed_en_dataset = en_temp_dataset.map(
        lambda examples: create_shuffled_sentence_pairs(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            add_special_tokens=True,
        ),
        batched=True,
        batch_size=64,
        remove_columns=en_temp_dataset.column_names
    )

    logger.info("Processing Arabic dataset")
    processed_ar_dataset = ar_temp_dataset.map(
        lambda examples: create_shuffled_sentence_pairs(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            add_special_tokens=True,
        ),
        batched=True,
        batch_size=64,
        remove_columns=ar_temp_dataset.column_names
    )

    # Split datasets into train and test
    test_size_ar = int(len(processed_ar_dataset) * args.test_split)
    test_size_en = int(len(processed_en_dataset) * args.test_split)

    test_ar_dataset = processed_ar_dataset.select(range(test_size_ar))
    test_en_dataset = processed_en_dataset.select(range(test_size_en))

    # Merge datasets
    test_dataset = concatenate_datasets([test_ar_dataset, test_en_dataset])
    merged_train_dataset = concatenate_datasets([processed_ar_dataset, processed_en_dataset]).shuffle(seed=args.seed)

    # Final datasets
    raw_datasets = {
        "train": merged_train_dataset,
        "test": test_dataset
    }

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
        min_lr_ratio=0.1  # Minimum learning rate will be 10% of max
    )

    # Create directory structure
    logger.info("Setting up directories")
    saving_directory = args.output_dir
    os.makedirs(saving_directory, exist_ok=True)

    # Train the model
    logger.info("Starting training")
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
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing value")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    # Dataset parameters
    parser.add_argument("--samples_per_lang", type=int, default=100, help="Number of samples per language")
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

    # Download NLTK resources if needed
    nltk.download('punkt')
    nltk.download('punkt_tab')

    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

    main(args)
