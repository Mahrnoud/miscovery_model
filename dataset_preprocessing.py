"""
Memory-efficient dataset preprocessing with streaming and chunked processing
"""
import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import gc

# Set up logging
logger = logging.getLogger(__name__)


class StreamingCSVDataset(Dataset):
    """
    Memory-efficient streaming dataset that processes CSV files on-demand
    """
    def __init__(self, csv_files, tokenizer, max_length=256, language_tags=None):
        self.csv_files = csv_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_tags = language_tags or {
            'en': '[LANG_EN]',
            'ar': '[LANG_AR]',
            'eg': '[LANG_AR_EG]'
        }

        # Build index of all samples across files
        self._build_index()

    def _build_index(self):
        """Build an index of sample locations without loading all data"""
        self.file_indices = []
        self.total_samples = 0

        logger.info("Building dataset index...")
        for csv_file in tqdm(self.csv_files, desc="Indexing files"):
            try:
                # Just count rows without loading full data
                df_info = pd.read_csv(csv_file, nrows=0)  # Just get columns
                if not all(col in df_info.columns for col in ['prompt', 'response', 'language']):
                    logger.warning(f"Skipping {csv_file}: Missing required columns")
                    continue

                # Count valid rows (this is still memory efficient)
                chunk_size = 1000
                valid_rows = 0
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                    # Remove invalid rows
                    chunk = chunk.dropna(subset=['prompt', 'response'])
                    chunk = chunk[chunk['prompt'].str.strip() != '']
                    chunk = chunk[chunk['response'].str.strip() != '']
                    valid_rows += len(chunk)

                self.file_indices.append({
                    'file': csv_file,
                    'start_idx': self.total_samples,
                    'count': valid_rows
                })
                self.total_samples += valid_rows

            except Exception as e:
                logger.error(f"Error indexing {csv_file}: {str(e)}")
                continue

        logger.info(f"Total samples indexed: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """Get a single sample by index"""
        # Find which file contains this index
        file_info = None
        for info in self.file_indices:
            if info['start_idx'] <= idx < info['start_idx'] + info['count']:
                file_info = info
                break

        if file_info is None:
            raise IndexError(f"Index {idx} out of range")

        # Calculate position within the file
        file_idx = idx - file_info['start_idx']

        # Load and process the specific row
        chunk_size = 1000
        current_valid_idx = 0

        for chunk in pd.read_csv(file_info['file'], chunksize=chunk_size):
            # Clean chunk
            chunk = chunk.dropna(subset=['prompt', 'response'])
            chunk = chunk[chunk['prompt'].str.strip() != '']
            chunk = chunk[chunk['response'].str.strip() != '']

            if current_valid_idx + len(chunk) > file_idx:
                # The target row is in this chunk
                row_in_chunk = file_idx - current_valid_idx
                row = chunk.iloc[row_in_chunk]

                # Process the row
                prompt = f"[REORDER] {self.language_tags.get(row['language'], '[LANG_UNKNOWN]')} {row['prompt']}"
                response = row['response']

                # Tokenize
                input_encoding = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True
                )
                target_encoding = self.tokenizer(
                    response,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True
                )

                return (
                    input_encoding["input_ids"].squeeze(0),
                    target_encoding["input_ids"].squeeze(0)
                )

            current_valid_idx += len(chunk)

        raise IndexError(f"Could not find index {idx} in file {file_info['file']}")


class ChunkedDatasetProcessor:
    """
    Process large datasets in chunks to avoid memory issues
    """
    def __init__(self, chunk_size=1000, cache_dir="./cache"):
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def process_csv_files_chunked(self, csv_files, tokenizer, max_length=256):
        """
        Process CSV files in chunks and save intermediate results
        """
        language_tags = {
            'en': '[LANG_EN]',
            'ar': '[LANG_AR]',
            'eg': '[LANG_AR_EG]'
        }

        chunk_files = []
        total_samples = 0

        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            logger.info(f"Processing {os.path.basename(csv_file)} in chunks")

            try:
                chunk_count = 0
                for chunk in pd.read_csv(csv_file, chunksize=self.chunk_size):
                    # Clean chunk
                    original_size = len(chunk)
                    chunk = chunk.dropna(subset=['prompt', 'response'])
                    chunk = chunk[chunk['prompt'].str.strip() != '']
                    chunk = chunk[chunk['response'].str.strip() != '']

                    if len(chunk) == 0:
                        continue

                    # Add language tags
                    chunk['prompt'] = chunk.apply(
                        lambda row: f"[REORDER] {language_tags.get(row['language'], '[LANG_UNKNOWN]')} {row['prompt']}",
                        axis=1
                    )

                    # Process chunk into tensors
                    input_tensors = []
                    target_tensors = []

                    for _, row in chunk.iterrows():
                        input_encoding = tokenizer(
                            row['prompt'],
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=True
                        )
                        target_encoding = tokenizer(
                            row['response'],
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=True
                        )
                        input_tensors.append(input_encoding["input_ids"].squeeze(0))
                        target_tensors.append(target_encoding["input_ids"].squeeze(0))

                    if input_tensors:
                        # Save chunk
                        chunk_file = os.path.join(
                            self.cache_dir,
                            f"chunk_{os.path.basename(csv_file)}_{chunk_count}.pt"
                        )
                        torch.save({
                            'input_ids': torch.stack(input_tensors),
                            'target_ids': torch.stack(target_tensors)
                        }, chunk_file)

                        chunk_files.append(chunk_file)
                        total_samples += len(input_tensors)
                        chunk_count += 1

                    # Clear memory
                    del chunk, input_tensors, target_tensors
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                continue

        logger.info(f"Processed {total_samples} samples into {len(chunk_files)} chunks")
        return chunk_files, total_samples


class ChunkedTensorDataset(Dataset):
    """
    Dataset that loads tensor chunks on demand
    """
    def __init__(self, chunk_files):
        self.chunk_files = chunk_files
        self.chunk_info = []
        self.total_samples = 0

        # Build index of chunks
        for chunk_file in chunk_files:
            try:
                data = torch.load(chunk_file, map_location='cpu')
                chunk_size = len(data['input_ids'])
                self.chunk_info.append({
                    'file': chunk_file,
                    'start_idx': self.total_samples,
                    'size': chunk_size
                })
                self.total_samples += chunk_size
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_file}: {e}")
                continue

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which chunk contains this index
        chunk_info = None
        for info in self.chunk_info:
            if info['start_idx'] <= idx < info['start_idx'] + info['size']:
                chunk_info = info
                break

        if chunk_info is None:
            raise IndexError(f"Index {idx} out of range")

        # Load chunk and get specific item
        data = torch.load(chunk_info['file'], map_location='cpu')
        chunk_idx = idx - chunk_info['start_idx']

        return (
            data['input_ids'][chunk_idx],
            data['target_ids'][chunk_idx]
        )


def prepare_memory_efficient_dataset(train_csv_directory, test_csv_directory=None,
                                   tokenizer=None, max_length=256, processing_method="streaming"):
    """
    Prepare datasets using memory-efficient methods

    Args:
        train_csv_directory: Directory with training CSV files
        test_csv_directory: Directory with test CSV files (optional)
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        processing_method: "streaming" or "chunked"

    Returns:
        Dictionary with train and test datasets
    """

    # Find CSV files
    train_csv_files = glob.glob(os.path.join(train_csv_directory, "*.csv"))
    test_csv_files = []
    if test_csv_directory:
        test_csv_files = glob.glob(os.path.join(test_csv_directory, "*.csv"))

    if not train_csv_files:
        logger.error(f"No CSV files found in {train_csv_directory}")
        return None

    logger.info(f"Found {len(train_csv_files)} training files")
    if test_csv_files:
        logger.info(f"Found {len(test_csv_files)} test files")

    if processing_method == "streaming":
        # Use streaming dataset (most memory efficient)
        train_dataset = StreamingCSVDataset(train_csv_files, tokenizer, max_length)

        if test_csv_files:
            test_dataset = StreamingCSVDataset(test_csv_files, tokenizer, max_length)
        else:
            # Split streaming dataset (more complex, simplified here)
            total_size = len(train_dataset)
            test_size = int(0.05 * total_size)
            train_size = total_size - test_size

            # Create indices for split
            indices = torch.randperm(total_size)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            # Note: This creates a subset, you might want to implement a proper split
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(train_dataset, test_indices)

    elif processing_method == "chunked":
        # Use chunked processing
        processor = ChunkedDatasetProcessor()

        # Process training files
        train_chunks, train_samples = processor.process_csv_files_chunked(
            train_csv_files, tokenizer, max_length
        )
        train_dataset = ChunkedTensorDataset(train_chunks)

        if test_csv_files:
            # Process test files
            test_chunks, test_samples = processor.process_csv_files_chunked(
                test_csv_files, tokenizer, max_length
            )
            test_dataset = ChunkedTensorDataset(test_chunks)
        else:
            # Split training data (simplified)
            # You would implement proper splitting logic here
            test_size = int(0.05 * len(train_dataset))
            train_size = len(train_dataset) - test_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, test_size]
            )

    return {
        "train": train_dataset,
        "test": test_dataset
    }


# Additional memory optimization functions
def clear_memory():
    """Clear Python and CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage():
    """Get current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return f"Memory usage: {memory_mb:.2f} MB"


# Drop-in replacement functions for backward compatibility
def prepare_high_quality_training_dataset_memory_efficient(train_csv_directory, test_csv_directory=None,
                                                          test_split_ratio=0.05, tokenizer=None,
                                                          max_seq_length=256, processing_method="streaming"):
    """
    Drop-in replacement for the original function with memory efficiency

    Args:
        train_csv_directory: Path to directory containing training CSV files
        test_csv_directory: Optional path to directory containing test CSV files
        test_split_ratio: Ratio for test split when test_csv_directory is None
        tokenizer: Tokenizer for processing (required for memory-efficient version)
        max_seq_length: Maximum sequence length
        processing_method: "streaming" or "chunked"

    Returns:
        Dictionary with train and test datasets (already tensorized)
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for memory-efficient processing")

    logger.info("Preparing memory-efficient datasets for fine-tuning")
    logger.info(get_memory_usage())

    datasets = prepare_memory_efficient_dataset(
        train_csv_directory=train_csv_directory,
        test_csv_directory=test_csv_directory,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        processing_method=processing_method
    )

    if datasets is None:
        logger.error("Failed to prepare datasets")
        return {"train": None, "test": None}

    # Handle test split if no separate test directory
    if test_csv_directory is None and processing_method == "streaming":
        # For streaming, we need to implement proper splitting
        total_size = len(datasets['train'])
        test_size = int(test_split_ratio * total_size)
        train_size = total_size - test_size

        # Create random indices for splitting
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Create subset datasets
        datasets['train'] = torch.utils.data.Subset(datasets['train'], train_indices)
        datasets['test'] = torch.utils.data.Subset(datasets['train'].dataset, test_indices)

    logger.info(f"Training dataset size: {len(datasets['train'])}")
    logger.info(f"Test dataset size: {len(datasets['test'])}")
    logger.info(get_memory_usage())

    # Clear memory after processing
    clear_memory()
    logger.info(f"After cleanup: {get_memory_usage()}")

    return datasets


def get_tensor_datasets_memory_efficient(raw_datasets, tokenizer, args):
    """
    Drop-in replacement that returns the datasets as-is since they're already tensorized

    Args:
        raw_datasets: Already processed tensor datasets
        tokenizer: Not used (kept for compatibility)
        args: Not used (kept for compatibility)

    Returns:
        The input datasets (already tensorized)
    """
    logger.info("Datasets are already tensorized - returning as-is")
    return raw_datasets


# Example usage function
def main_memory_efficient(train_dir, test_dir=None, tokenizer=None):
    """
    Main function demonstrating memory-efficient processing
    """
    logger.info("Starting memory-efficient dataset preparation")
    logger.info(get_memory_usage())

    # Use streaming for maximum memory efficiency
    datasets = prepare_memory_efficient_dataset(
        train_csv_directory=train_dir,
        test_csv_directory=test_dir,
        tokenizer=tokenizer,
        max_length=256,
        processing_method="streaming"  # or "chunked"
    )

    logger.info(f"Training dataset size: {len(datasets['train'])}")
    logger.info(f"Test dataset size: {len(datasets['test'])}")
    logger.info(get_memory_usage())

    # Clear memory after processing
    clear_memory()
    logger.info(f"After cleanup: {get_memory_usage()}")

    return datasets
