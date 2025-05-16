"""
Dataset Preprocessing Module for Transformer-based models.
This module handles all dataset loading, preprocessing, and preparation.
"""
import os
import random
import re
import logging
import pandas as pd
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import TensorDataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
import nltk

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_SEQ_LENGTH = 128
MIN_WORDS_FOR_SHUFFLE = 4


# Helper functions for text processing
def is_english(text):
    """
    Determine if text is in English using language detection.
    """
    try:
        sample_text = text[:500].replace("[LANG_EN]", "").replace("[LANG_AR]", "").strip()
        if not sample_text:
            return any(ord(char) < 128 for char in text)
        return detect(sample_text) == 'en'
    except LangDetectException:
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        total_chars = len(text)
        if total_chars == 0:
            return True
        return (ascii_chars / total_chars) > 0.5


def shuffle_words_in_text(text):
    """
    Shuffle words in text to create training pairs.
    Returns the shuffled text and a boolean indicating if shuffling occurred.
    """
    words = text.split()
    if len(words) < MIN_WORDS_FOR_SHUFFLE:
        return text, False
    shuffled_words = words[:]
    random.shuffle(shuffled_words)
    if shuffled_words == words and len(words) > 1:
        idx1, idx2 = random.sample(range(len(words)), 2)
        shuffled_words[idx1], shuffled_words[idx2] = shuffled_words[idx2], shuffled_words[idx1]
    return " ".join(shuffled_words), True


def create_shuffled_sentence_pairs(examples, tokenizer, max_length=MAX_SEQ_LENGTH, add_special_tokens=True):
    """
    Create pairs of shuffled and original sentences for training.
    """
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


def create_tensor_datasets(prompts, responses, tokenizer, max_length=256, cache_file=None):
    """
    Convert prompts and responses to tensor datasets for model training.
    If cache_file is provided, will attempt to load from or save to cache.
    """
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


def clean_text(text):
    """
    Clean text by replacing special characters.
    """
    if isinstance(text, str):
        return re.sub(r'\r\n', ' ', text)
    return text


# Functions for finetuning dataset
def is_within_limit(example, tokenizer, max_len=500):
    """
    Check if example is within token length limit.
    """
    en_total = tokenizer(example["prompt_en"], example["response_en"], truncation=False, add_special_tokens=True)
    ar_total = tokenizer(example["prompt_ar"], example["response_ar"], truncation=False, add_special_tokens=True)
    return len(en_total["input_ids"]) <= max_len and len(ar_total["input_ids"]) <= max_len


def to_prompt_response(example):
    """
    Convert example to prompt/response format for both languages.
    """
    return {
        "prompt_en": clean_text(f"{example['en_question']}"),
        "response_en": clean_text(example["en_answer"]),
        "prompt_ar": clean_text(f"{example['ar_question']}"),
        "response_ar": clean_text(example["ar_answer"])
    }


def flatten_bilingual_dataset(dataset):
    """
    Flatten bilingual dataset to a list of prompts/responses.
    """
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


# Main preprocessing functions
def prepare_training_dataset(tokenizer, args):
    """
    Prepare dataset for initial training with sentence shuffling approach.

    Args:
        tokenizer: Tokenizer to use for encoding
        args: Arguments containing preprocessing parameters

    Returns:
        Dictionary with train and test datasets
    """
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, nltk.downloader.DownloadError):
        nltk.download('punkt')
        nltk.download('punkt_tab')

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

    return raw_datasets


def prepare_finetuning_dataset(tokenizer, args):
    """
    Prepare dataset for finetuning on math reasoning task.

    Args:
        tokenizer: Tokenizer to use for encoding
        args: Arguments containing preprocessing parameters

    Returns:
        Dictionary with train and test datasets
    """
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

    # Apply transformation
    transformed_ds = ds.map(to_prompt_response)
    transformed_csv = csv_dataset.map(to_prompt_response)

    # Combine and filter by token length
    combined = concatenate_datasets([transformed_ds, transformed_csv])
    filtered_combined = combined.filter(lambda x: is_within_limit(x, tokenizer))

    # Flatten to prompt/response pairs
    final_dataset = flatten_bilingual_dataset(filtered_combined)

    # Shuffle and split
    final_dataset = final_dataset.shuffle(seed=args.seed)
    val_size = max(1, int(0.01 * len(final_dataset)))
    val_dataset = final_dataset.select(range(val_size))
    train_dataset = final_dataset.select(range(val_size, len(final_dataset)))

    # Final structure
    raw_datasets = {
        "train": train_dataset,
        "test": val_dataset
    }

    return raw_datasets


def get_tensor_datasets(raw_datasets, tokenizer, args):
    """
    Convert raw datasets to tensor datasets ready for model training.

    Args:
        raw_datasets: Dictionary containing 'train' and 'test' datasets
        tokenizer: Tokenizer to use for encoding
        args: Arguments containing preprocessing parameters

    Returns:
        Dictionary with train and test tensor datasets
    """
    cache_dir = args.cache_dir
    train_cache_file = f"{cache_dir}/train_tensors_{args.max_seq_length}.pt"
    test_cache_file = f"{cache_dir}/test_tensors_{args.max_seq_length}.pt"

    os.makedirs(cache_dir, exist_ok=True)

    train_dataset = create_tensor_datasets(
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

    tensor_datasets = {
        "train": train_dataset,
        "test": test_dataset
    }

    return tensor_datasets
