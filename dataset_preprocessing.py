"""
Enhanced Dataset Preprocessing Module for Transformer-based models.
This module handles all dataset loading, preprocessing, and preparation
with advanced quality control and cleaning functionality.
"""
import os
import random
import re
import logging
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import TensorDataset
from tqdm import tqdm
from langdetect import detect, LangDetectException
import nltk
from collections import Counter

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_SEQ_LENGTH = 128
MIN_WORDS_FOR_SHUFFLE = 4


# ========================== DATASET INSPECTION FUNCTIONS ==========================

def inspect_dataset_statistics(dataset, name="Dataset"):
    """
    Print statistics about a dataset to understand its composition.
    """
    total_examples = len(dataset)

    # Sample a few examples to analyze
    sample_size = min(1000, total_examples)
    samples = dataset.select(range(sample_size))

    # Get the key field (article or text)
    text_key = 'article' if 'article' in samples.features else 'text'

    # Analyze text lengths
    text_lengths = [len(ex[text_key]) for ex in samples if ex[text_key]]

    if not text_lengths:
        logger.warning(f"{name}: No valid texts found in sample!")
        return

    # Calculate statistics
    avg_length = sum(text_lengths) / len(text_lengths)
    min_length = min(text_lengths)
    max_length = max(text_lengths)

    logger.info(f"=== {name} Statistics ===")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Average text length: {avg_length:.1f} characters")
    logger.info(f"Min text length: {min_length} characters")
    logger.info(f"Max text length: {max_length} characters")
    logger.info(f"Examples with empty text: {sample_size - len(text_lengths)}")

    # Distribution of lengths
    bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]
    bin_names = ["0-100", "100-500", "500-1000", "1000-5000", "5000-10000", "10000+"]
    distribution = [0] * len(bin_names)

    for length in text_lengths:
        for i, upper in enumerate(bins[1:]):
            if length < upper:
                distribution[i] += 1
                break

    logger.info("Text length distribution:")
    for name, count in zip(bin_names, distribution):
        percentage = (count / len(text_lengths)) * 100
        logger.info(f"  {name}: {count} examples ({percentage:.1f}%)")


def inspect_sample_examples(dataset, num_samples=5, text_key='article'):
    """
    Print sample examples from the dataset for manual inspection.
    """
    logger.info(f"\n=== Sample Examples from Dataset ({text_key}) ===")

    samples = dataset.shuffle().select(range(min(num_samples, len(dataset))))

    for i, sample in enumerate(samples):
        text = sample[text_key]
        if not isinstance(text, str):
            logger.warning(f"Example {i+1}: Invalid text type: {type(text)}")
            continue

        logger.info(f"\nExample {i+1}:")
        logger.info(f"Length: {len(text)} characters")

        # Print the first 500 characters to get a sense of the content
        preview = text[:500] + "..." if len(text) > 500 else text
        logger.info(f"Preview: {preview}")

        # If there are prompts and responses, show those too
        if 'prompt' in sample and 'response' in sample:
            logger.info(f"\nPrompt: {sample['prompt'][:200]}...")
            logger.info(f"Response: {sample['response'][:200]}...")

        logger.info("-" * 80)


# ========================== DATA CLEANING AND FILTERING FUNCTIONS ==========================

def clean_text(text):
    """
    Clean text by replacing special characters and normalize formatting.
    """
    if not isinstance(text, str):
        return ""

    # Replace newlines and returns with spaces
    text = re.sub(r'\r\n|\r|\n', ' ', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove other common noise patterns
    text = re.sub(r'\[\s*cite\s*\]', '', text)

    return text


def clean_and_filter_dataset(dataset, text_key='article', min_length=100, max_length=20000):
    """
    Clean and filter dataset examples based on quality criteria.
    Returns filtered dataset and count of removed examples.
    """
    original_size = len(dataset)
    logger.info(f"Starting dataset cleaning with {original_size} examples")

    # Define quality filter functions
    def has_minimum_length(example):
        text = example[text_key]
        if not isinstance(text, str) or not text.strip():
            return False
        return len(text) >= min_length

    def has_reasonable_length(example):
        text = example[text_key]
        return len(text) <= max_length

    def has_good_sentence_count(example):
        """Check if text has a reasonable number of sentences."""
        text = example[text_key]
        try:
            sentences = nltk.sent_tokenize(text)
            return 3 <= len(sentences) <= 300  # Adjust thresholds as needed
        except:
            return False

    def has_good_sentence_lengths(example):
        """Check if sentences have reasonable lengths."""
        text = example[text_key]
        try:
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                return False
            avg_length = sum(len(s) for s in sentences) / len(sentences)
            return 20 <= avg_length <= 200  # Adjust thresholds as needed
        except:
            return False

    def has_balanced_punctuation(example):
        """Check if text has reasonable punctuation."""
        text = example[text_key]
        # Check for unusual punctuation patterns
        punct_count = sum(1 for c in text if c in '.,;:?!')
        text_length = len(text)

        if text_length == 0:
            return False

        punct_ratio = punct_count / text_length
        return 0.01 <= punct_ratio <= 0.2  # Adjust thresholds as needed

    def no_excessive_special_chars(example):
        """Check for unusual special character patterns."""
        text = example[text_key]
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:?!\'"-()[]{}')
        text_length = len(text)

        if text_length == 0:
            return False

        special_ratio = special_chars / text_length
        return special_ratio <= 0.1  # Adjust threshold as needed

    # Clean text function for mapping
    def clean_text_content(example):
        text = example[text_key]
        example[text_key] = clean_text(text)
        return example

    # Apply text cleaning
    logger.info("Cleaning text content...")
    cleaned_dataset = dataset.map(clean_text_content)

    # Apply quality filters
    logger.info("Applying quality filters...")
    logger.info("Filter 1: Minimum length")
    filtered_dataset = cleaned_dataset.filter(has_minimum_length)
    logger.info(f"  Remaining: {len(filtered_dataset)}/{original_size} examples")

    logger.info("Filter 2: Maximum length")
    filtered_dataset = filtered_dataset.filter(has_reasonable_length)
    logger.info(f"  Remaining: {len(filtered_dataset)}/{original_size} examples")

    logger.info("Filter 3: Sentence count")
    filtered_dataset = filtered_dataset.filter(has_good_sentence_count)
    logger.info(f"  Remaining: {len(filtered_dataset)}/{original_size} examples")

    logger.info("Filter 4: Sentence lengths")
    filtered_dataset = filtered_dataset.filter(has_good_sentence_lengths)
    logger.info(f"  Remaining: {len(filtered_dataset)}/{original_size} examples")

    logger.info("Filter 5: Punctuation balance")
    filtered_dataset = filtered_dataset.filter(has_balanced_punctuation)
    logger.info(f"  Remaining: {len(filtered_dataset)}/{original_size} examples")

    logger.info("Filter 6: Special character ratio")
    filtered_dataset = filtered_dataset.filter(no_excessive_special_chars)
    logger.info(f"  Remaining: {len(filtered_dataset)}/{original_size} examples")

    # Calculate removed examples
    final_size = len(filtered_dataset)
    removed_count = original_size - final_size
    removal_percentage = (removed_count / original_size) * 100 if original_size > 0 else 0

    logger.info(f"Filtered out {removed_count} examples ({removal_percentage:.1f}%)")
    logger.info(f"Original size: {original_size}, Final size: {final_size}")

    return filtered_dataset


# ========================== LANGUAGE-SPECIFIC QUALITY CHECKS ==========================

def check_english_quality(example, text_key='article'):
    """Additional quality checks specific to English texts."""
    import re

    text = example[text_key]
    if not isinstance(text, str):
        return False

    # Check for reasonable word count
    words = text.split()
    if len(words) < 30:
        return False

    # Check for sentence capitalization (majority of sentences should start with capital)
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 3:
        return False

    capital_sentence_count = sum(1 for s in sentences if s and s[0].isalpha() and s[0].isupper())
    capital_ratio = capital_sentence_count / len(sentences) if sentences else 0
    if capital_ratio < 0.6:  # At least 60% should start with capital
        return False

    # Check for reasonable word length
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    if not (3 <= avg_word_length <= 12):
        return False

    # Check for excessive all-caps text
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = caps_words / len(words) if words else 0
    if caps_ratio > 0.3:  # No more than 30% all-caps words
        return False

    return True


def check_arabic_quality(example, text_key='text'):
    """Additional quality checks specific to Arabic texts."""
    import re

    text = example[text_key]
    if not isinstance(text, str):
        return False

    # Check for Arabic character presence
    arabic_char_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    arabic_chars = sum(1 for c in text if arabic_char_pattern.match(c or ''))
    if arabic_chars / max(1, len(text)) < 0.5:  # At least 50% Arabic characters
        return False

    # Check for reasonable word count
    words = text.split()
    if len(words) < 30:
        return False

    # Check for reasonable structure
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 3:
        return False

    # Check for punctuation in Arabic text
    arabic_punct = set(['،', '؛', '؟', '!', '.'])
    punct_count = sum(1 for c in text if c in arabic_punct)
    if punct_count < 3:  # Need some minimal punctuation
        return False

    return True


# ========================== CONTENT QUALITY CHECKS ==========================

def detect_template_or_boilerplate(example, text_key='article'):
    """
    Detect texts that are likely templates or boilerplate content.
    """
    text = example[text_key]
    if not isinstance(text, str):
        return True

    # Check for common template patterns
    template_patterns = [
        r'{{.*?}}',
        r'\[\[.*?\]\]',
        r'<template.*?>',
        r'the following is a transcript',  # Common for transcripts
        r'this article is a stub',  # Wikipedia stub marker
    ]

    for pattern in template_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Check for repeated paragraph structures
    paragraphs = text.split('\n\n')
    if len(paragraphs) >= 3:
        # Check first few words of paragraphs for similarities
        para_starts = [p.split()[:3] for p in paragraphs if p and len(p.split()) >= 3]
        if len(para_starts) >= 3:
            # If more than half of paragraph starts are identical, likely a template
            starts_text = [' '.join(start).lower() for start in para_starts]
            most_common = max(set(starts_text), key=starts_text.count)
            if starts_text.count(most_common) / len(starts_text) > 0.5:
                return True

    return False


def detect_low_information_content(example, text_key='article'):
    """
    Detect texts with low information density.
    """
    text = example[text_key]
    if not isinstance(text, str):
        return True

    words = text.split()

    if len(words) < 50:
        return True

    # Calculate lexical diversity (unique words / total words)
    unique_words = len(set(w.lower() for w in words))
    diversity = unique_words / len(words) if words else 0

    # Low diversity might indicate repetitive content
    if diversity < 0.4:  # Adjust threshold as needed
        return True

    # Check for repetitive sentences
    sentences = nltk.sent_tokenize(text)
    if len(sentences) >= 5:
        # Create sentence fingerprints (first 5 words)
        sentence_starts = [' '.join(s.split()[:5]).lower() for s in sentences if len(s.split()) >= 5]
        if len(sentence_starts) >= 5:
            # Count duplicates
            counts = Counter(sentence_starts)
            max_repeat = counts.most_common(1)[0][1] if counts else 0

            # If any sentence start appears too frequently, flag it
            if max_repeat > 3 or max_repeat / len(sentence_starts) > 0.3:
                return True

    return False


# ========================== BASIC PREPROCESSING FUNCTIONS ==========================

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
            response_tokens = tokenizer(response_str, max_length=max_length, truncation=True, padding=False)["input_ids"]

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


# ========================== FUNCTIONS FOR FINETUNING DATASET ==========================

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


# ========================== MAIN DATASET PREPARATION FUNCTIONS ==========================

def prepare_high_quality_training_dataset(tokenizer, args):
    """
    Prepare high-quality dataset for training with enhanced filtering and cleaning.

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
    # Load the datasets with a larger sample to allow for filtering
    sample_multiplier = 2  # Load more samples since we'll filter some out

    # For CNN/DailyMail, load a larger sample
    en_dataset_stream = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    en_samples = list(en_dataset_stream.take(args.samples_per_lang * sample_multiplier))
    en_temp_dataset = Dataset.from_dict(
        {'article': [sample['article'] for sample in en_samples if sample.get('article')]})

    # For XLSum, load a larger sample
    ar_dataset_stream = load_dataset("csebuetnlp/xlsum", "arabic", split="train", streaming=True)
    ar_samples = list(ar_dataset_stream.take(args.samples_per_lang * sample_multiplier))
    ar_temp_dataset = Dataset.from_dict({'text': [sample['text'] for sample in ar_samples if sample.get('text')]})

    # Inspect datasets before cleaning
    logger.info("Inspecting datasets before cleaning:")
    inspect_dataset_statistics(en_temp_dataset, "CNN/DailyMail (English)")
    inspect_dataset_statistics(ar_temp_dataset, "XLSum (Arabic)")

    # Show sample examples before cleaning
    inspect_sample_examples(en_temp_dataset, num_samples=2, text_key='article')
    inspect_sample_examples(ar_temp_dataset, num_samples=2, text_key='text')

    # Clean and filter datasets
    logger.info("Cleaning and filtering English dataset...")
    cleaned_en_dataset = clean_and_filter_dataset(en_temp_dataset, text_key='article')

    logger.info("Cleaning and filtering Arabic dataset...")
    cleaned_ar_dataset = clean_and_filter_dataset(ar_temp_dataset, text_key='text')

    # Apply language-specific filters
    logger.info("Applying language-specific quality filters to English dataset...")
    high_quality_en = cleaned_en_dataset.filter(
        lambda ex: check_english_quality(ex, text_key='article') and
                  not detect_template_or_boilerplate(ex, text_key='article') and
                  not detect_low_information_content(ex, text_key='article')
    )

    logger.info("Applying language-specific quality filters to Arabic dataset...")
    high_quality_ar = cleaned_ar_dataset.filter(
        lambda ex: check_arabic_quality(ex, text_key='text') and
                  not detect_template_or_boilerplate(ex, text_key='text') and
                  not detect_low_information_content(ex, text_key='text')
    )

    # Inspect datasets after cleaning
    logger.info("Inspecting datasets after cleaning:")
    inspect_dataset_statistics(high_quality_en, "Cleaned CNN/DailyMail (English)")
    inspect_dataset_statistics(high_quality_ar, "Cleaned XLSum (Arabic)")

    # Show sample examples after cleaning
    inspect_sample_examples(high_quality_en, num_samples=2, text_key='article')
    inspect_sample_examples(high_quality_ar, num_samples=2, text_key='text')

    # Sample from cleaned datasets
    NUM_SAMPLES_PER_LANG = args.samples_per_lang
    logger.info(f"Sampling {NUM_SAMPLES_PER_LANG} examples from each cleaned language dataset")

    # Either sample randomly or take the first N examples
    if len(high_quality_en) > NUM_SAMPLES_PER_LANG:
        high_quality_en = high_quality_en.shuffle(seed=args.seed).select(range(NUM_SAMPLES_PER_LANG))
    else:
        logger.warning(f"Only {len(high_quality_en)} English examples after filtering, wanted {NUM_SAMPLES_PER_LANG}")

    if len(high_quality_ar) > NUM_SAMPLES_PER_LANG:
        high_quality_ar = high_quality_ar.shuffle(seed=args.seed).select(range(NUM_SAMPLES_PER_LANG))
    else:
        logger.warning(f"Only {len(high_quality_ar)} Arabic examples after filtering, wanted {NUM_SAMPLES_PER_LANG}")

    # Process datasets to create shuffled sentence pairs
    logger.info("Processing English dataset")
    processed_en_dataset = high_quality_en.map(
        lambda examples: create_shuffled_sentence_pairs(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            add_special_tokens=True,
        ),
        batched=True,
        batch_size=64,
        remove_columns=high_quality_en.column_names
    )

    logger.info("Processing Arabic dataset")
    processed_ar_dataset = high_quality_ar.map(
        lambda examples: create_shuffled_sentence_pairs(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            add_special_tokens=True,
        ),
        batched=True,
        batch_size=64,
        remove_columns=high_quality_ar.column_names
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

    # Log final dataset statistics
    logger.info("Final dataset statistics:")
    logger.info(f"Training examples: {len(raw_datasets['train'])}")
    logger.info(f"Testing examples: {len(raw_datasets['test'])}")

    # Sample some prompt/response pairs for inspection
    if len(raw_datasets['train']) > 0:
        samples = raw_datasets['train'].shuffle(seed=args.seed).select(range(min(5, len(raw_datasets['train']))))
        logger.info("\n=== Sample Training Prompt/Response Pairs ===")
        for i, sample in enumerate(samples):
            logger.info(f"\nPair {i+1}:")
            logger.info(f"Prompt: {sample['prompt']}")
            logger.info(f"Response: {sample['response']}")
            logger.info("-" * 40)

    return raw_datasets


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
