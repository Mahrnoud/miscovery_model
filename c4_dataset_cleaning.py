import re
import pandas as pd
from collections import defaultdict
import gc
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class FastArabicTextCleaner:
    """Optimized Arabic text cleaner for Google Colab"""

    def __init__(self, tokenizer_name="miscovery/tokenizer_v2", max_tokens=256):
        # Initialize tokenizer with optimizations
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,  # Use fast tokenizer if available
                padding_side="right"
            )
            print(f"✅ Loaded tokenizer: {tokenizer_name}")

            # Pre-encode some test text to warm up the tokenizer
            _ = self.tokenizer.encode("تجربة", add_special_tokens=True)

        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            self.tokenizer = None

        self.max_tokens = max_tokens
        self.stats = defaultdict(int)

        # Compile all patterns once
        self._compile_optimized_patterns()

        # Create character translation table for invisible chars
        invisible_chars = "\u200C\u200D\u200E\u200F\u202A\u202B\u202C\u202D\u202E\u00AD\u2028\u009C\u200b\uFEFF"
        self.invisible_trans = str.maketrans('', '', invisible_chars)

    def _compile_optimized_patterns(self):
        """Compile all regex patterns with optimization flags"""

        # Combine all URL patterns into one mega-pattern for single pass
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            r'\b(?:Goal\.com|coptology\.com|Uptobox\.com|Mediafile\.co|10shared\.com|Vidto\.me|3rbup\.com|Openload\.co|1tube\.to|Allmyvideos\.net|2shared\.com|arabic\.alibaba\.com|Releases\.ae)\b[^\s]*',
            r'//[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*',
            r'\S+\.(html?|php|asp|jsp|rar|zip|exe|pdf|doc|docx|txt|mp3|mp4|avi)\b',
            r'[&%][0-9A-Fa-f]{2}[A-Za-z0-9%&=_-]{10,}',
            r'\b(?:https?|ftp|ftps):\s*(?!\w)',  # Leftover protocols
            r'File\s+size\s+\d+\s+bytes'
        ]
        self.all_urls = re.compile('|'.join(f'({pattern})' for pattern in url_patterns), re.IGNORECASE)

        # Combine datetime patterns
        datetime_patterns = [
            r'\d{2}-\d{2}-\d{4},\s+\d{2}:\d{2}\s+(?:PM|AM)',
            r'(?:الأحد|الإثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت)\s+\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4}\s+\d{2}:\d{2}\s+(?:صباحا|مساء)',
            r'آخر\s+حضور\s*[»›]\s*\d{2}-\d{2}-\d{4}\s*\(\d{2}:\d{2}\s+(?:AM|PM)\)',
            r'T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}',
            r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}'
        ]
        self.all_datetime = re.compile('|'.join(f'({pattern})' for pattern in datetime_patterns), re.IGNORECASE)

        # Combine forum patterns
        forum_patterns = [
            r'\d{2}-\d{2}-\d{4},\s+\d{2}:\d{2}\s+(AM|PM)\s+رقم المشاركة\s*:\s*\d+',
            r'آخر مشاركة:\s*\d{2}-\d{2}-\d{4},\s+\d{2}:\d{2}\s+(AM|PM)',
            r'\[\s*قراءة:\s*\d+\s*\|\s*طباعة:\s*\d+\s*\|\s*إرسال لصديق:\s*\d+\s*\]',
            r'مشاركتي في اليوم بمعدل:\s*[\d.]+',
            r'مشاهدة النسخة كاملة\s*[:：]?',
            r'آخر تعديل بتاريخ\s+\d+\s+\w+\s+\d{4}،?\s*في\s+\d{2}:\d{2}'
        ]
        self.all_forum = re.compile('|'.join(f'({pattern})' for pattern in forum_patterns), re.IGNORECASE)

        # Other optimized patterns
        self.all_tags = re.compile(r'<[^>]+>|\[color=[^]]*\]|\[/color:[^]]*\]|\[/color\]|\[[^\]]*\]', re.IGNORECASE)
        self.emojis = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]+')
        self.emails_phones = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|[\d\u0660-\u0669]{3,}[-\s\d\u0660-\u0669]{7,}')
        self.repeated_chars = re.compile(r'([!@#$%^&*()_+=\-\[\]{}|\\:";\'<>?,.~/`])\1{3,}')
        self.excessive_punct = re.compile(r'[.]{4,}|[،]{2,}|[؟]{2,}|[؛]{2,}')
        self.multiple_spaces = re.compile(r'\s+')
        self.social_concat = re.compile(
            r'\b(?:Facebook\d*Twitter|TwitterGoogle\+|Google\+ReddIt|ReddItWhatsApp|WhatsAppPinterest|FacebookTwitterGoogle\+ReddItWhatsAppPinterest)\b',
            re.IGNORECASE)
        self.arabic_no_spaces = re.compile(r'[\u0600-\u06FF]{50,}')
        self.navigation = re.compile(
            r'(الرئيسية|الأخبار|اتصل بنا|من نحن|الخصوصية|الشروط|جميع الحقوق محفوظة|©|copyright|all rights reserved)\s*[/|>]?',
            re.IGNORECASE)

    def count_tokens_fast(self, text: str) -> int:
        """Fast token counting with early termination"""
        if not self.tokenizer or not text.strip():
            return 0

        try:
            # Use truncation for faster processing
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_tokens + 50  # Small buffer for accurate counting
            )
            return len(tokens)
        except:
            # Fast fallback: approximate token count
            return len(text.split())

    def clean_text_fast(self, text: str) -> str:
        """Ultra-fast text cleaning with combined patterns"""
        if not text or not isinstance(text, str):
            return ""

        # Remove invisible chars first (fastest operation)
        text = text.translate(self.invisible_trans)

        # Replace newlines with spaces
        text = text.replace('\n', ' ')

        # Apply all major pattern removals in order of impact
        text = self.all_urls.sub('', text)  # Remove all URLs/files
        text = self.all_datetime.sub('', text)  # Remove datetime patterns
        text = self.all_forum.sub('', text)  # Remove forum patterns
        text = self.all_tags.sub('', text)  # Remove all tags
        text = self.emojis.sub('', text)  # Remove emojis
        text = self.emails_phones.sub('', text)  # Remove emails/phones
        text = self.social_concat.sub('', text)  # Remove social concatenations
        text = self.arabic_no_spaces.sub('', text)  # Remove long Arabic concatenations
        text = self.navigation.sub('', text)  # Remove navigation

        # Fix repeated characters
        text = self.repeated_chars.sub(r'\1', text)

        # Normalize punctuation
        text = self.excessive_punct.sub(lambda m: m.group(0)[0], text)

        # Normalize spaces (final step)
        text = self.multiple_spaces.sub(' ', text).strip()

        return text


def process_colab_optimized(dataset, num_samples=10000, skip_samples=0, save_csv=True,
                            csv_min_length=30, max_tokens=256,
                            tokenizer_name="miscovery/tokenizer_v2",
                            chunk_size=500, save_frequency=2000):
    """
    Colab-optimized processing with memory management and progress saving

    Args:
        dataset: HuggingFace streaming dataset
        num_samples: Number of samples to process
        skip_samples: Number of samples to skip at the beginning
        save_csv: Whether to save CSV files
        csv_min_length: Minimum text length for CSV
        max_tokens: Maximum token limit
        tokenizer_name: Tokenizer name
        chunk_size: Process in chunks (smaller = less memory)
        save_frequency: Save progress every N samples
    """

    print(f"🚀 Colab-Optimized Processing: {num_samples:,} samples (skipping first {skip_samples:,})")
    print(f"📦 Chunk size: {chunk_size} | 💾 Save frequency: {save_frequency}")

    # Initialize cleaner
    cleaner = FastArabicTextCleaner(tokenizer_name=tokenizer_name, max_tokens=max_tokens)

    # Results storage
    all_results = []
    stats = defaultdict(int)

    # Progress tracking
    processed_count = 0
    skipped_count = 0
    within_limit_count = 0
    total_original_chars = 0
    total_cleaned_chars = 0

    # Create progress bar for total samples to process (including skipped ones)
    total_to_iterate = skip_samples + num_samples
    pbar = tqdm(total=total_to_iterate, desc="Processing",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    chunk_buffer = []

    try:
        for i, sample in enumerate(dataset):
            if i >= total_to_iterate:
                break

            # Skip the first skip_samples records
            if i < skip_samples:
                skipped_count += 1
                if skipped_count % 10000 == 0:  # Update progress every 10K skipped
                    pbar.update(10000)
                    pbar.set_postfix({'status': f'skipping... ({skipped_count:,}/{skip_samples:,})'})
                continue

            # Update progress for the final skip batch
            if skipped_count == skip_samples and len(chunk_buffer) == 0:
                remaining_skip = skip_samples % 10000
                if remaining_skip > 0:
                    pbar.update(remaining_skip)
                pbar.set_postfix({'status': 'processing...'})

            chunk_buffer.append(sample)

            # Process chunk when buffer is full
            if len(chunk_buffer) >= chunk_size:
                # Process the chunk
                chunk_results = process_chunk_fast(chunk_buffer, cleaner, csv_min_length, max_tokens)

                # Update statistics
                for result in chunk_results:
                    if result:  # Valid result
                        all_results.append(result)
                        stats['processed'] += 1
                        if result['within_limit']:
                            within_limit_count += 1
                        total_original_chars += result['original_length']
                        total_cleaned_chars += result['cleaned_length']

                processed_count += len(chunk_buffer)

                # Update progress bar
                pbar.update(len(chunk_buffer))
                reduction_pct = (1 - total_cleaned_chars / max(1, total_original_chars)) * 100
                pbar.set_postfix({
                    'valid': len(all_results),
                    'reduction': f"{reduction_pct:.1f}%",
                    'within_limit': f"{within_limit_count}/{processed_count}"
                })

                # Save progress periodically
                if processed_count % save_frequency == 0 and all_results:
                    save_progress_csv([r['prompt'] for r in all_results],
                                      processed_count + skip_samples, max_tokens,
                                      prefix=f"batch_{skip_samples // 1000}k_to_{(skip_samples + processed_count) // 1000}k")

                    # Memory cleanup
                    gc.collect()

                # Clear chunk buffer
                chunk_buffer = []

        # Process remaining samples
        if chunk_buffer:
            chunk_results = process_chunk_fast(chunk_buffer, cleaner, csv_min_length, max_tokens)
            for result in chunk_results:
                if result:
                    all_results.append(result)
                    if result['within_limit']:
                        within_limit_count += 1
                    total_original_chars += result['original_length']
                    total_cleaned_chars += result['cleaned_length']

            processed_count += len(chunk_buffer)
            pbar.update(len(chunk_buffer))

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")

    finally:
        pbar.close()

    # Final save with batch info in filename
    if all_results and save_csv:
        detailed_file, simple_file = save_final_csv_with_batch(
            all_results, processed_count, max_tokens, skip_samples
        )

        # Print final statistics
        print(f"\n📊 FINAL RESULTS (Batch {skip_samples // 1000}K-{(skip_samples + processed_count) // 1000}K):")
        print(f"✅ Skipped samples: {skip_samples:,}")
        print(f"✅ Processed samples: {processed_count:,}")
        print(f"✅ Valid prompts: {len(all_results):,}")
        print(f"✅ Within token limit: {within_limit_count:,} ({within_limit_count / len(all_results) * 100:.1f}%)")

        if total_original_chars > 0:
            reduction_pct = (1 - total_cleaned_chars / total_original_chars) * 100
            print(f"✅ Size reduction: {reduction_pct:.1f}%")
            print(f"✅ Avg chars/prompt: {total_cleaned_chars // len(all_results):,}")

        print(f"💾 Detailed CSV: {detailed_file}")
        print(f"💾 Simple CSV: {simple_file}")

        return all_results, detailed_file, simple_file

    return all_results, None, None


def process_chunk_fast(chunk, cleaner, min_length, max_tokens):
    """Process a chunk of samples quickly"""
    results = []

    for sample in chunk:
        try:
            original_text = sample.get('text', '')
            if len(original_text) < 10:  # Skip very short texts early
                continue

            # Clean text
            cleaned_text = cleaner.clean_text_fast(original_text)

            # Early filtering
            if len(cleaned_text) < min_length:
                continue

            # Count tokens
            token_count = cleaner.count_tokens_fast(cleaned_text)
            within_limit = token_count <= max_tokens

            results.append({
                'prompt': cleaned_text,
                'original_length': len(original_text),
                'cleaned_length': len(cleaned_text),
                'token_count': token_count,
                'within_limit': within_limit
            })

        except Exception as e:
            # Skip problematic samples
            continue

    return results


def save_progress_csv(prompts, processed_count, max_tokens, prefix="progress"):
    """Save progress CSV file"""
    filename = f"/content/{prefix}_cleaned_arabic_{processed_count}_max{max_tokens}tokens.csv"

    try:
        df = pd.DataFrame({'prompt': prompts})
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n💾 Progress saved: {len(prompts):,} prompts -> {filename}")
    except Exception as e:
        print(f"\n❌ Error saving progress: {e}")

    return filename


def save_final_csv(results_data, processed_count, max_tokens):
    """Save final CSV file with detailed info including token counts"""

    # Filter only results that are within token limit
    within_limit_results = [item for item in results_data if item['within_limit']]

    # Save detailed CSV with all metadata (only within limit)
    detailed_filename = f"/content/final_cleaned_arabic_{processed_count}_max_{max_tokens}_tokens_detailed.csv"

    # Save simple CSV with just prompts (for training) - only within limit
    simple_filename = f"/content/final_cleaned_arabic_{processed_count}_max_{max_tokens}_tokens.csv"

    try:
        # Create detailed DataFrame with only within-limit results
        detailed_df = pd.DataFrame(within_limit_results)
        detailed_df.to_csv(detailed_filename, index=False, encoding='utf-8')

        # Create simple DataFrame with just prompts (only within limit)
        simple_df = pd.DataFrame({'prompt': [item['prompt'] for item in within_limit_results]})
        simple_df.to_csv(simple_filename, index=False, encoding='utf-8')

        # Create summary file
        summary_filename = detailed_filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Arabic Text Cleaning Summary\n")
            f.write(f"===========================\n")
            f.write(f"Total prompts (within limit): {len(within_limit_results):,}\n")
            f.write(f"Total processed samples: {len(results_data):,}\n")
            f.write(f"Filtered out (over limit): {len(results_data) - len(within_limit_results):,}\n")
            f.write(
                f"Average character length: {sum(item['cleaned_length'] for item in within_limit_results) // max(1, len(within_limit_results)):,} chars\n")
            f.write(
                f"Average token count: {sum(item['token_count'] for item in within_limit_results) // max(1, len(within_limit_results)):.1f} tokens\n")
            f.write(f"All prompts within token limit: {len(within_limit_results):,}\n")
            f.write(f"Original processed samples: {processed_count:,}\n")
            f.write(f"Final success rate: {len(within_limit_results) / processed_count * 100:.1f}%\n")
            f.write(f"\nFiles created:\n")
            f.write(f"- Detailed CSV: {detailed_filename}\n")
            f.write(f"- Simple CSV: {simple_filename}\n")

        print(f"✅ Detailed CSV saved: {detailed_filename} ({len(within_limit_results):,} within-limit prompts)")
        print(f"✅ Simple CSV saved: {simple_filename} ({len(within_limit_results):,} within-limit prompts)")
        print(f"📋 Summary saved: {summary_filename}")
        print(f"🗑️ Filtered out {len(results_data) - len(within_limit_results):,} prompts that exceeded token limit")

        return detailed_filename, simple_filename

    except Exception as e:
        print(f"❌ Error saving final CSV: {e}")
        return None, None


def save_final_csv_with_batch(results_data, processed_count, max_tokens, skip_samples):
    """Save final CSV file with batch information in filename"""

    # Filter only results that are within token limit
    within_limit_results = [item for item in results_data if item['within_limit']]

    # Create batch-specific filenames
    start_batch = skip_samples // 1000
    end_batch = (skip_samples + processed_count) // 1000
    batch_suffix = f"batch_{start_batch}k_to_{end_batch}k"

    detailed_filename = f"/content/final_cleaned_arabic_{batch_suffix}_max_{max_tokens}_tokens_detailed.csv"
    simple_filename = f"/content/final_cleaned_arabic_{batch_suffix}_max_{max_tokens}_tokens.csv"

    try:
        # Create detailed DataFrame with only within-limit results
        detailed_df = pd.DataFrame(within_limit_results)
        detailed_df.to_csv(detailed_filename, index=False, encoding='utf-8')

        # Create simple DataFrame with just prompts (only within limit)
        simple_df = pd.DataFrame({'prompt': [item['prompt'] for item in within_limit_results]})
        simple_df.to_csv(simple_filename, index=False, encoding='utf-8')

        # Create summary file
        summary_filename = detailed_filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Arabic Text Cleaning Summary - Batch {start_batch}K-{end_batch}K\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Batch range: Records {skip_samples:,} to {skip_samples + processed_count:,}\n")
            f.write(f"Total prompts (within limit): {len(within_limit_results):,}\n")
            f.write(f"Total processed samples: {len(results_data):,}\n")
            f.write(f"Filtered out (over limit): {len(results_data) - len(within_limit_results):,}\n")
            f.write(
                f"Average character length: {sum(item['cleaned_length'] for item in within_limit_results) // max(1, len(within_limit_results)):,} chars\n")
            f.write(
                f"Average token count: {sum(item['token_count'] for item in within_limit_results) // max(1, len(within_limit_results)):.1f} tokens\n")
            f.write(f"Success rate: {len(within_limit_results) / processed_count * 100:.1f}%\n")
            f.write(f"\nFiles created:\n")
            f.write(f"- Detailed CSV: {detailed_filename}\n")
            f.write(f"- Simple CSV: {simple_filename}\n")

        print(f"✅ Detailed CSV saved: {detailed_filename} ({len(within_limit_results):,} within-limit prompts)")
        print(f"✅ Simple CSV saved: {simple_filename} ({len(within_limit_results):,} within-limit prompts)")
        print(f"📋 Summary saved: {summary_filename}")

        return detailed_filename, simple_filename

    except Exception as e:
        print(f"❌ Error saving final CSV: {e}")
        return None, None


def quick_test():
    """Quick test of the optimized cleaner"""
    print("🧪 Testing optimized cleaner...")

    test_texts = [
        "هذا نص تجريبي https://www.example.com مع رابط 💐 ورموز تعبيرية ✅",
        "يمكنك زيارة Goal.com أو coptology.com للمزيد http: و https: protocols",
        "05-07-2013, 12:52 PM رقم المشاركة: 123 مع تاريخ قديم",
        "Facebook773TwitterGoogle+ReddItWhatsApp social concatenation"
    ]

    cleaner = FastArabicTextCleaner()

    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean_text_fast(text)
        tokens = cleaner.count_tokens_fast(cleaned)
        print(f"\nTest {i}:")
        print(f"  Original: {text}")
        print(f"  Cleaned:  {cleaned}")
        print(f"  Tokens:   {tokens}")


# Memory-efficient dataset loading for Colab
def load_dataset_colab_friendly():
    """Load dataset with Colab optimizations"""
    print("📡 Loading Arabic C4 dataset (streaming mode)...")

    try:
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            verification_mode='no_checks'  # Skip verification for speed
        )
        print("✅ Dataset loaded successfully")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


# Example usage for Google Colab
if __name__ == "__main__":
    print("🔥 COLAB-OPTIMIZED ARABIC TEXT CLEANER")
    print("=" * 50)

    # Quick test first
    quick_test()

    # Load dataset
    dataset = load_dataset_colab_friendly()

    if dataset:
        print("\n🚀 Starting optimized processing...")

        # For testing (small batch)
        print("\n=== SMALL TEST (1000 samples) ===")
        # Process records 100,000 to 200,000 (next 100K)
        results, detailed_file, simple_file = process_colab_optimized(
            dataset=dataset,
            num_samples=500000,  # Process 100K samples
            skip_samples=300000,  # Skip first 100K samples
            save_csv=True,
            csv_min_length=30,
            max_tokens=250,
            chunk_size=1000,
            save_frequency=5000000
        )

        # For larger processing (adjust as needed)
        print("\n=== PRODUCTION SETTINGS ===")
        print("# For larger batches, use:")
        print("""
        results, detailed_file, simple_file = process_colab_optimized(
            dataset=dataset,
            num_samples=50000,          # Larger batch
            save_csv=True,
            csv_min_length=50,
            max_tokens=256,
            chunk_size=300,             # Balanced for Colab
            save_frequency=2000         # Save every 2000 samples
        )
        """)

        print("\n📁 OUTPUT FILES EXPLANATION:")
        print("✅ *_detailed.csv - Contains: prompt, original_length, cleaned_length, token_count, within_limit")
        print("✅ *.csv - Contains: prompt only (for training)")
        print("✅ *_summary.txt - Contains: statistics and file info")

        print("\n💡 COLAB OPTIMIZATION TIPS:")
        print("✅ Use chunk_size=200-500 for Colab's memory limits")
        print("✅ Set save_frequency=1000-2000 to save progress")
        print("✅ Monitor RAM usage in Colab's sidebar")
        print("✅ If memory issues occur, reduce chunk_size to 100")
        print("✅ Resume processing from the last saved progress file")

    else:
        print("❌ Could not load dataset. Check your internet connection.")
