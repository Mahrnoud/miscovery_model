"""
Enhanced dataset preprocessing with separate train/test directories
"""
import os
import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from datasets import Dataset
import glob

# Set up logging
logger = logging.getLogger(__name__)


def load_and_process_csv_directory(csv_directory):
    """
    Load all CSV files from a directory, clean the data, and add language tags.

    Args:
        csv_directory: Path to directory containing CSV files

    Returns:
        Combined dataset with language tags added to prompts
    """
    # Language tag mapping
    language_tags = {
        'en': '[LANG_EN]',
        'ar': '[LANG_AR]',
        'eg': '[LANG_AR_EG]'
    }

    all_data = []
    total_removed_rows = 0
    file_stats = {}

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in directory: {csv_directory}")
        return Dataset.from_dict({"prompt": [], "response": []})

    logger.info(f"Found {len(csv_files)} CSV files to process in {csv_directory}")

    for csv_file in tqdm(csv_files, desc=f"Processing CSV files from {os.path.basename(csv_directory)}"):
        logger.info(f"Processing file: {os.path.basename(csv_file)}")

        try:
            # Load CSV file
            df = pd.read_csv(csv_file)
            original_rows = len(df)

            # Check if required columns exist
            required_columns = ['prompt', 'response', 'language']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Skipping {csv_file}: Missing columns {missing_columns}")
                continue

            # Remove rows where prompt or response is empty/null
            initial_count = len(df)
            df = df.dropna(subset=['prompt', 'response'])
            df = df[df['prompt'].str.strip() != '']
            df = df[df['response'].str.strip() != '']

            removed_count = initial_count - len(df)
            total_removed_rows += removed_count

            # Add language tags to prompts
            df['prompt'] = df.apply(
                lambda row: f"[REORDER] {language_tags.get(row['language'], '[LANG_UNKNOWN]')} {row['prompt']}",
                axis=1
            )

            # Keep only prompt and response columns
            processed_df = df[['prompt', 'response']].copy()
            all_data.append(processed_df)

            # Store statistics for this file
            file_stats[os.path.basename(csv_file)] = {
                'original_rows': original_rows,
                'removed_rows': removed_count,
                'final_rows': len(processed_df)
            }

            logger.info(f"  Original rows: {original_rows}")
            logger.info(f"  Removed rows: {removed_count}")
            logger.info(f"  Final rows: {len(processed_df)}")

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            continue

    if not all_data:
        logger.warning(f"No valid data found in any CSV files from {csv_directory}")
        return Dataset.from_dict({"prompt": [], "response": []})

    # Merge all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(merged_df)

    # Log final statistics
    logger.info("\n" + "=" * 50)
    logger.info(f"PROCESSING SUMMARY for {os.path.basename(csv_directory)}")
    logger.info("=" * 50)

    for filename, stats in file_stats.items():
        logger.info(f"{filename}:")
        logger.info(f"  Original rows: {stats['original_rows']}")
        logger.info(f"  Removed rows: {stats['removed_rows']}")
        logger.info(f"  Final rows: {stats['final_rows']}")
        logger.info("-" * 30)

    logger.info(f"Total removed rows across all files: {total_removed_rows}")
    logger.info(f"Final merged dataset size: {len(dataset)}")
    logger.info("=" * 50)

    # Print summary to console as well
    print("\n" + "=" * 50)
    print(f"PROCESSING SUMMARY for {os.path.basename(csv_directory)}")
    print("=" * 50)

    for filename, stats in file_stats.items():
        print(f"{filename}:")
        print(f"  Original rows: {stats['original_rows']}")
        print(f"  Removed rows: {stats['removed_rows']}")
        print(f"  Final rows: {stats['final_rows']}")
        print("-" * 30)

    print(f"Total removed rows across all files: {total_removed_rows}")
    print(f"Final merged dataset size: {len(dataset)}")
    print("=" * 50)

    return dataset


def prepare_high_quality_training_dataset(train_csv_directory, test_csv_directory=None, test_split_ratio=0.05):
    """
    Load CSV files from directories and prepare training datasets.

    Args:
        train_csv_directory: Path to directory containing training CSV files
        test_csv_directory: Optional path to directory containing test CSV files. 
                          If None, will split from training data using test_split_ratio
        test_split_ratio: Ratio of data to use for testing when test_csv_directory is None (default: 0.05)

    Returns:
        Dictionary with train and test datasets
    """
    # Load and process training CSV files
    logger.info("Loading training dataset...")
    train_dataset = load_and_process_csv_directory(train_csv_directory)

    if len(train_dataset) == 0:
        logger.warning("No training data loaded from CSV files")
        return {"train": train_dataset, "test": Dataset.from_dict({"prompt": [], "response": []})}

    # Handle test dataset
    if test_csv_directory is not None:
        # Load test data from separate directory
        logger.info("Loading test dataset from separate directory...")
        test_dataset = load_and_process_csv_directory(test_csv_directory)

        if len(test_dataset) == 0:
            logger.warning("No test data loaded from CSV files, falling back to splitting training data")
            # Fall back to splitting training data
            split_datasets = train_dataset.train_test_split(test_size=test_split_ratio, seed=42)
            train_dataset = split_datasets['train']
            test_dataset = split_datasets['test']
        else:
            logger.info(f"Successfully loaded {len(test_dataset)} test examples from {test_csv_directory}")
            # Use all training data since we have separate test data
            # No need to split train_dataset
    else:
        # Split training data into train and test
        logger.info(f"Splitting training data with test ratio: {test_split_ratio}")
        split_datasets = train_dataset.train_test_split(test_size=test_split_ratio, seed=42)
        train_dataset = split_datasets['train']
        test_dataset = split_datasets['test']

    # Final datasets
    raw_datasets = {
        "train": train_dataset,
        "test": test_dataset
    }

    # Log final dataset statistics
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Training examples: {len(raw_datasets['train'])}")
    logger.info(f"Testing examples: {len(raw_datasets['test'])}")

    if test_csv_directory is not None:
        logger.info(f"Test data source: Separate directory ({test_csv_directory})")
    else:
        logger.info(f"Test data source: Split from training data ({test_split_ratio * 100:.1f}%)")

    print(f"\nTraining examples: {len(raw_datasets['train'])}")
    print(f"Testing examples: {len(raw_datasets['test'])}")

    # Sample some prompt/response pairs for inspection
    if len(raw_datasets['train']) > 0:
        samples = raw_datasets['train'].shuffle(seed=42).select(range(min(5, len(raw_datasets['train']))))
        logger.info("\n=== Sample Training Prompt/Response Pairs ===")
        print("\n=== Sample Training Prompt/Response Pairs ===")
        for i, sample in enumerate(samples):
            logger.info(f"\nPair {i + 1}:")
            logger.info(f"Prompt: {sample['prompt']}")
            logger.info(f"Response: {sample['response']}")
            logger.info("-" * 40)
            print(f"Pair {i + 1}:")
            print(f"Prompt: {sample['prompt']}")
            print(f"Response: {sample['response']}")
            print("-" * 40)

    # Also show test samples if available
    if len(raw_datasets['test']) > 0:
        test_samples = raw_datasets['test'].shuffle(seed=42).select(range(min(3, len(raw_datasets['test']))))
        logger.info("\n=== Sample Test Prompt/Response Pairs ===")
        print("\n=== Sample Test Prompt/Response Pairs ===")
        for i, sample in enumerate(test_samples):
            logger.info(f"\nTest Pair {i + 1}:")
            logger.info(f"Prompt: {sample['prompt']}")
            logger.info(f"Response: {sample['response']}")
            logger.info("-" * 40)
            print(f"Test Pair {i + 1}:")
            print(f"Prompt: {sample['prompt']}")
            print(f"Response: {sample['response']}")
            print("-" * 40)

    logger.info("=" * 60)
    return raw_datasets


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
