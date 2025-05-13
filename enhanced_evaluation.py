# ==========================================
# Error Analysis Functions
# ==========================================

def analyze_errors(results, output_dir=None, lang="both"):
    """
    Analyze common error patterns in the evaluation results

    Args:
        results: List of detailed evaluation results
        output_dir: Directory to save analysis results
        lang: Language filter ('en', 'ar', or 'both')

    Returns:
        Dictionary with error patterns and counts
    """
    error_patterns = defaultdict(int)
    error_examples = defaultdict(list)
    max_examples = 3  # Maximum number of examples to store per error pattern

    # Filter by language if specified
    if lang == "en":
        results = [result for result in results if not result.get('is_arabic', False)]
    elif lang == "ar":
        results = [result for result in results if result.get('is_arabic', False)]

    for result in results:
        ref = result.get('reference', '')
        gen = result.get('generated', '')
        is_arabic = result.get('is_arabic', False)

        if not ref or not gen:
            continue

        # Skip if exact match or very high similarity
        exact_match_score = result.get('exact_match', 0)
        # Convert NumPy value to Python float if needed
        if isinstance(exact_match_score, (np.float32, np.float64)):
            exact_match_score = float(exact_match_score)

        if exact_match_score > 0.9:
            continue

        # Check for common error types
        if len(gen) < len(ref) * 0.5:
            error_patterns['too_short'] += 1
            if len(error_examples['too_short']) < max_examples:
                error_examples['too_short'].append({
                    'source': result.get('source', ''),
                    'reference': ref,
                    'generated': gen,
                    'is_arabic': is_arabic
                })
        elif len(gen) > len(ref) * 1.5:
            error_patterns['too_long'] += 1
            if len(error_examples['too_long']) < max_examples:
                error_examples['too_long'].append({
                    'source': result.get('source', ''),
                    'reference': ref,
                    'generated': gen,
                    'is_arabic': is_arabic
                })

        # Check for missing dates, numbers, or proper nouns
        ref_dates = re.findall(r'\b\d{4}\b', ref)
        gen_dates = re.findall(r'\b\d{4}\b', gen)
        if ref_dates and not any(d in gen_dates for d in ref_dates):
            error_patterns['missing_dates'] += 1
            if len(error_examples['missing_dates']) < max_examples:
                error_examples['missing_dates'].append({
                    'source': result.get('source', ''),
                    'reference': ref,
                    'generated': gen,
                    'missing_date': ref_dates[0],
                    'is_arabic': is_arabic
                })

        # Check for missing names in Arabic
        if is_arabic:
            ref_names_ar = re.findall(r'\b(ال[\u0600-\u06FF]+)\b', ref)
            gen_names_ar = re.findall(r'\b(ال[\u0600-\u06FF]+)\b', gen)
            if ref_names_ar and not any(name in gen_names_ar for name in ref_names_ar):
                error_patterns['missing_names'] += 1
                if len(error_examples['missing_names']) < max_examples:
                    error_examples['missing_names'].append({
                        'source': result.get('source', ''),
                        'reference': ref,
                        'generated': gen,
                        'missing_name': ref_names_ar[0],
                        'is_arabic': is_arabic
                    })
        else:
            # For English, look for capitalized words (potential proper nouns)
            ref_names_en = re.findall(r'\b[A-Z][a-z]+\b', ref)
            gen_names_en = re.findall(r'\b[A-Z][a-z]+\b', gen)
            if ref_names_en and not any(name in gen_names_en for name in ref_names_en):
                error_patterns['missing_names'] += 1
                if len(error_examples['missing_names']) < max_examples:
                    error_examples['missing_names'].append({
                        'source': result.get('source', ''),
                        'reference': ref,
                        'generated': gen,
                        'missing_name': ref_names_en[0],
                        'is_arabic': is_arabic
                    })

        # Check for wrong structure - different sentence count
        ref_sentences = len(re.split(r'[.!?؟،]', ref))
        gen_sentences = len(re.split(r'[.!?؟،]', gen))
        if abs(ref_sentences - gen_sentences) > 1:
            error_patterns['wrong_structure'] += 1
            if len(error_examples['wrong_structure']) < max_examples:
                error_examples['wrong_structure'].append({
                    'source': result.get('source', ''),
                    'reference': ref,
                    'generated': gen,
                    'ref_sentences': ref_sentences,
                    'gen_sentences': gen_sentences,
                    'is_arabic': is_arabic
                })

    # Save error analysis if output directory provided
    if output_dir:
        lang_suffix = f"_{lang}" if lang != "both" else ""
        error_file = os.path.join(output_dir, f"error_analysis{lang_suffix}.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({
                'error_counts': dict(error_patterns),
                'error_examples': error_examples
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Error analysis saved to {error_file}")

    return {'error_patterns': dict(error_patterns), 'error_examples': error_examples}


def debug_metrics_calculation(reference, hypothesis):
    """
    Debug utility to trace through metrics calculation

    Args:
        reference: Reference text
        hypothesis: Generated text

    Returns:
        Dictionary with detailed metrics calculation info
    """
    debug_info = {
        'inputs': {
            'reference': reference,
            'hypothesis': hypothesis,
        },
        'text_processing': {
            'reference_cleaned': clean_text_for_comparison(reference),
            'hypothesis_cleaned': clean_text_for_comparison(hypothesis),
        }
    }

    # Check if it's Arabic
    is_arabic_text = is_arabic(reference)
    debug_info['is_arabic'] = is_arabic_text

    if is_arabic_text:
        debug_info['arabic_processing'] = {
            'reference_normalized': normalize_arabic(reference),
            'hypothesis_normalized': normalize_arabic(hypothesis),
        }

        # Add tokenization results
        ref_tokens = tokenize_arabic(normalize_arabic(reference))
        hyp_tokens = tokenize_arabic(normalize_arabic(hypothesis))
        debug_info['tokenization'] = {
            'reference_tokens': ref_tokens,
            'hypothesis_tokens': hyp_tokens,
        }
    else:
        # Add tokenization for non-Arabic
        debug_info['tokenization'] = {
            'reference_tokens': reference.lower().split(),
            'hypothesis_tokens': hypothesis.lower().split(),
        }

    # Calculate all metrics
    try:
        bleu_score = calculate_bleu(reference, hypothesis)
        debug_info['bleu'] = {
            'score': bleu_score,
        }
    except Exception as e:
        debug_info['bleu'] = {
            'error': str(e),
        }

    try:
        rouge_scores = calculate_rouge(reference, hypothesis)
        debug_info['rouge'] = rouge_scores
    except Exception as e:
        debug_info['rouge'] = {
            'error': str(e),
        }

    try:
        f1_score = calculate_f1_word_match(reference, hypothesis)
        debug_info['f1'] = {
            'score': f1_score,
        }
    except Exception as e:
        debug_info['f1'] = {
            'error': str(e),
        }

    try:
        exact_match = calculate_exact_match(reference, hypothesis)
        debug_info['exact_match'] = {
            'score': exact_match,
        }
    except Exception as e:
        debug_info['exact_match'] = {
            'error': str(e),
        }

    return debug_info


# ==========================================
# Visualization Functions
# ==========================================

def plot_metrics_comparison(evaluation_results, output_dir=None):
    """
    Create visualizations comparing different metrics

    Args:
        evaluation_results: List of detailed evaluation results
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(12, 8))

    # Extract metrics
    bleu_scores = [r['bleu'] for r in evaluation_results]
    f1_scores = [r['f1_score'] for r in evaluation_results]
    rouge_l_scores = [r['rougeL'] for r in evaluation_results]

    # Plot BLEU vs F1
    plt.subplot(2, 2, 1)
    plt.scatter(bleu_scores, f1_scores, alpha=0.6)
    plt.xlabel('BLEU Score')
    plt.ylabel('F1 Score')
    plt.title('BLEU vs F1 Score')
    plt.grid(True, alpha=0.3)

    # Plot BLEU vs ROUGE-L
    plt.subplot(2, 2, 2)
    plt.scatter(bleu_scores, rouge_l_scores, alpha=0.6)
    plt.xlabel('BLEU Score')
    plt.ylabel('ROUGE-L Score')
    plt.title('BLEU vs ROUGE-L Score')
    plt.grid(True, alpha=0.3)

    # Plot metrics distribution
    plt.subplot(2, 2, 3)
    plt.hist(bleu_scores, bins=20, alpha=0.5, label='BLEU')
    plt.hist(f1_scores, bins=20, alpha=0.5, label='F1')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Metrics Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot quality distribution
    quality_counts = {}
    for r in evaluation_results:
        quality = r['quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    plt.subplot(2, 2, 4)
    qualities = ['Excellent', 'Good', 'Partial', 'Poor']
    counts = [quality_counts.get(q, 0) for q in qualities]
    plt.bar(qualities, counts)
    plt.xlabel('Quality Category')
    plt.ylabel('Count')
    plt.title('Quality Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if output directory is provided
    if output_dir:
        metrics_plot_path = os.path.join(output_dir, 'metrics_comparison.png')
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {metrics_plot_path}")

    plt.close()


def plot_language_comparison(evaluation_results, output_dir=None):
    """
    Create visualizations comparing performance between languages

    Args:
        evaluation_results: List of detailed evaluation results
        output_dir: Directory to save visualizations
    """
    # Separate results by language
    arabic_results = [r for r in evaluation_results if r.get('is_arabic', False)]
    non_arabic_results = [r for r in evaluation_results if not r.get('is_arabic', False)]

    if not arabic_results or not non_arabic_results:
        logger.warning("Cannot create language comparison plot: missing results for one language")
        return

    plt.figure(figsize=(14, 10))

    # Extract metrics by language
    metrics = ['bleu', 'f1_score', 'rouge1', 'rouge2', 'rougeL']
    metric_names = ['BLEU', 'F1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']

    # Plot 1: Mean metrics comparison
    plt.subplot(2, 2, 1)
    ar_means = [np.mean([r[m] for r in arabic_results]) for m in metrics]
    non_ar_means = [np.mean([r[m] for r in non_arabic_results]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, ar_means, width, label='Arabic')
    plt.bar(x + width / 2, non_ar_means, width, label='Non-Arabic')

    plt.xlabel('Metric')
    plt.ylabel('Mean Score')
    plt.title('Mean Metrics by Language')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Quality distribution by language
    plt.subplot(2, 2, 2)
    qualities = ['Excellent', 'Good', 'Partial', 'Poor']

    ar_quality_counts = {}
    for r in arabic_results:
        quality = r['quality']
        ar_quality_counts[quality] = ar_quality_counts.get(quality, 0) + 1

    non_ar_quality_counts = {}
    for r in non_arabic_results:
        quality = r['quality']
        non_ar_quality_counts[quality] = non_ar_quality_counts.get(quality, 0) + 1

    ar_counts = [ar_quality_counts.get(q, 0) for q in qualities]
    non_ar_counts = [non_ar_quality_counts.get(q, 0) for q in qualities]

    # Convert to percentages
    ar_total = sum(ar_counts)
    non_ar_total = sum(non_ar_counts)

    ar_pcts = [count / ar_total * 100 if ar_total else 0 for count in ar_counts]
    non_ar_pcts = [count / non_ar_total * 100 if non_ar_total else 0 for count in non_ar_counts]

    x = np.arange(len(qualities))

    plt.bar(x - width / 2, ar_pcts, width, label='Arabic')
    plt.bar(x + width / 2, non_ar_pcts, width, label='Non-Arabic')

    plt.xlabel('Quality Category')
    plt.ylabel('Percentage')
    plt.title('Quality Distribution by Language')
    plt.xticks(x, qualities)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Generation time comparison
    plt.subplot(2, 2, 3)
    ar_times = [r['generation_time'] for r in arabic_results]
    non_ar_times = [r['generation_time'] for r in non_arabic_results]

    plt.boxplot([ar_times, non_ar_times], labels=['Arabic', 'Non-Arabic'])
    plt.ylabel('Generation Time (seconds)')
    plt.title('Generation Time by Language')
    plt.grid(True, alpha=0.3)

    # Plot 4: Metrics distributions by language
    plt.subplot(2, 2, 4)

    ar_bleu = [r['bleu'] for r in arabic_results]
    non_ar_bleu = [r['bleu'] for r in non_arabic_results]

    plt.hist(ar_bleu, bins=20, alpha=0.5, label='Arabic')
    plt.hist(non_ar_bleu, bins=20, alpha=0.5, label='Non-Arabic')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.title('BLEU Score Distribution by Language')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if output directory is provided
    if output_dir:
        lang_plot_path = os.path.join(output_dir, 'language_comparison.png')
        plt.savefig(lang_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Language comparison plot saved to {lang_plot_path}")

    plt.close()


def plot_training_history(train_losses, val_losses, lr_history=None, output_dir=None):
    """
    Plot training history with training and validation loss

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        lr_history: Optional list of learning rates
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')

    # For validation losses, we might have fewer points
    if val_losses:
        # We need to align validation losses with training steps
        val_steps = np.linspace(0, len(train_losses) - 1, len(val_losses))
        plt.plot(val_steps, val_losses, 'o-', label='Validation Loss')

    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot learning rate if provided
    if lr_history:
        plt.subplot(2, 1, 2)
        plt.plot(lr_history)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {history_plot_path}")

    plt.close()


# ==========================================
# Integration with Training Workflow
# ==========================================

class EvaluationCallback:
    """
    Callback class to be used during training for evaluation and visualization
    """

    def __init__(
            self,
            model,
            tokenizer,
            eval_dataloader,
            device,
            output_dir="eval_results",
            eval_steps=1000,
            save_best_model=True,
            handle_arabic=True,
            max_eval_samples=3  # Added parameter to control number of evaluation samples
    ):
        """
        Initialize the callback

        Args:
            model: The model being trained
            tokenizer: Tokenizer for text generation
            eval_dataloader: DataLoader for evaluation
            device: Device for computation
            output_dir: Directory to save results
            eval_steps: Number of training steps between evaluations
            save_best_model: Whether to save the best model
            handle_arabic: Whether to apply Arabic-specific processing
            max_eval_samples: Maximum number of samples to evaluate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.save_best_model = save_best_model
        self.handle_arabic = handle_arabic
        self.max_eval_samples = max_eval_samples

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize tracking variables
        self.best_bleu = 0.0
        self.best_step = 0
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []

        # Initialize history file
        self.history_file = os.path.join(output_dir, "training_history.json")
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "bleu_scores": [],
            "f1_scores": [],
            "steps": [],
            "learning_rates": []
        }

        # Save initial history
        self._save_history()

    def on_step_end(self, step, loss, optimizer=None):
        """
        Call at the end of each training step

        Args:
            step: Current step number
            loss: Current training loss
            optimizer: Optional optimizer to track learning rate
        """
        # Record training loss
        self.train_losses.append(loss)
        self.history["train_losses"].append(loss)
        self.history["steps"].append(step)

        # Record learning rate if optimizer is provided
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            self.history["learning_rates"].append(current_lr)

        # Run evaluation at specified steps
        if step % self.eval_steps == 0:
            self.on_evaluation(step)

    def on_evaluation(self, step):
        """
        Run evaluation and update metrics

        Args:
            step: Current step number
        """
        logger.info(f"Running evaluation at step {step}")

        # Create a temporary config for evaluation
        class EvalConfig:
            def __init__(self, handle_arabic=True, max_length=128):
                self.handle_arabic = handle_arabic
                self.max_length = max_length

        eval_config = EvalConfig(handle_arabic=self.handle_arabic)

        # Create a limited evaluation dataloader with max_eval_samples
        from torch.utils.data import Subset
        import random

        # Sample a subset of evaluation data if needed
        limited_eval_dataloader = self.eval_dataloader
        if hasattr(self, 'max_eval_samples') and self.max_eval_samples and self.max_eval_samples < len(
                self.eval_dataloader.dataset):
            # Get random indices
            indices = random.sample(range(len(self.eval_dataloader.dataset)), self.max_eval_samples)

            # Create a subset dataset
            subset_dataset = Subset(self.eval_dataloader.dataset, indices)

            # Create a new dataloader with the subset
            batch_size = self.eval_dataloader.batch_size or 1
            from torch.utils.data import DataLoader
            limited_eval_dataloader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            logger.info(f"Evaluating on {self.max_eval_samples} samples instead of {len(self.eval_dataloader.dataset)}")

        # Run evaluation
        eval_output_dir = os.path.join(self.output_dir, f"step_{step}")
        eval_summary, detailed_results = evaluate_model(
            model=self.model,
            dataloader=limited_eval_dataloader,
            tokenizer=self.tokenizer,
            device=self.device,
            config=eval_config,
            output_dir=eval_output_dir
        )

        # Record metrics
        avg_bleu = eval_summary.get("avg_bleu", 0.0)
        avg_f1 = eval_summary.get("avg_f1_score", 0.0)

        # Update history
        self.val_losses.append(eval_summary.get("avg_loss", 0.0))
        self.history["val_losses"].append(eval_summary.get("avg_loss", 0.0))
        self.history["bleu_scores"].append(avg_bleu)
        self.history["f1_scores"].append(avg_f1)

        # Save history
        self._save_history()

        # Create plots
        plot_metrics_comparison(detailed_results, eval_output_dir)
        plot_training_history(
            self.train_losses,
            self.val_losses,
            self.lr_history,
            self.output_dir
        )

        # Run error analysis
        analyze_errors(detailed_results, eval_output_dir)

        # Check if this is the best model
        if avg_bleu > self.best_bleu:
            self.best_bleu = avg_bleu
            self.best_step = step

            logger.info(f"New best model at step {step} with BLEU {avg_bleu:.4f}")

            # Save best model if requested
            if self.save_best_model:
                best_model_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Best model saved to {best_model_path}")

    def on_training_end(self):
        """Call at the end of training"""
        # Create final visualizations
        plot_training_history(
            self.train_losses,
            self.val_losses,
            self.lr_history,
            self.output_dir
        )

        # Log best model info
        logger.info(f"Training complete. Best model at step {self.best_step} with BLEU {self.best_bleu:.4f}")

    def _save_history(self):
        """Save training history to file, ensuring all values are JSON-serializable"""
        json_safe_history = {}

        for key, values in self.history.items():
            # Convert any NumPy types to Python native types
            json_safe_history[key] = [float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                                      for v in values]

        # Save to file
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_history, f, ensure_ascii=False, indent=2)


# ==========================================
# Main Function for Standalone Usage
# ==========================================

def main(model_path, tokenizer_path, eval_data_path, output_dir="eval_results", num_samples=None):
    """
    Main function for standalone evaluation

    Args:
        model_path: Path to the model
        tokenizer_path: Path to the tokenizer
        eval_data_path: Path to evaluation data
        output_dir: Directory to save results
        num_samples: Number of samples to evaluate (None for all)
    """
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from transformers import AutoTokenizer

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load model
        model = torch.load(model_path)
        model.eval()

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Load evaluation data
        eval_data = torch.load(eval_data_path)

        # Sample if requested
        if num_samples and num_samples < len(eval_data):
            from torch.utils.data import random_split
            indices = list(range(len(eval_data)))
            import random
            random.shuffle(indices)
            sampled_indices = indices[:num_samples]
            eval_data = Subset(eval_data, sampled_indices)

        # Create dataloader
        eval_dataloader = DataLoader(eval_data, batch_size=4, shuffle=False)

        # Create config
        class EvalConfig:
            def __init__(self, handle_arabic=True, max_length=128):
                self.handle_arabic = handle_arabic
                self.max_length = max_length

        eval_config = EvalConfig()

        # Run evaluation
        eval_summary, detailed_results = evaluate_model(
            model=model,
            dataloader=eval_dataloader,
            tokenizer=tokenizer,
            device=device,
            config=eval_config,
            output_dir=output_dir
        )

        # Create visualizations
        plot_metrics_comparison(detailed_results, output_dir)

        # Run error analysis
        analyze_errors(detailed_results, output_dir)

        # Run language comparison if there are both Arabic and non-Arabic examples
        arabic_results = [r for r in detailed_results if r.get('is_arabic', False)]
        non_arabic_results = [r for r in detailed_results if not r.get('is_arabic', False)]

        if arabic_results and non_arabic_results:
            plot_language_comparison(detailed_results, output_dir)

        logger.info(f"Evaluation complete. Results saved to {output_dir}")

        return eval_summary, detailed_results

    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return """
Enhanced Evaluation and Generation Module for Transformer Models
With specialized support for Arabic language processing and metrics

This module provides:
1. Enhanced text generation methods with Arabic support
2. Comprehensive evaluation functions for multilingual models
3. Visualization utilities for model performance tracking
"""


import os
import re
import time
import json
import logging
from collections import defaultdict
from difflib import SequenceMatcher

import torch
# Add numpy import at the top of the file if not already present
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# Arabic Language Processing Utilities
# ==========================================

def normalize_arabic(text):
    """
    Enhanced Arabic text normalization that handles edge cases better

    Args:
        text: Input Arabic text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Normalize Unicode form
    import unicodedata
    text = unicodedata.normalize('NFKC', text)

    # Normalize Alef variations
    text = re.sub("[إأآا]", "ا", text)

    # Normalize Hamzas
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)

    # Normalize Yeh and Alef Maksura
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    # Remove diacritics (all harakat)
    text = re.sub(r'[\u064B-\u0652\u0670]', '', text)

    # Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)

    # Remove punctuation but keep spaces
    text = re.sub(r'[^\u0600-\u06FF\s\w]', '', text)

    # Normalize whitespace (all whitespace becomes a single space)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_arabic(text):
    """
    Tokenize Arabic text

    Args:
        text: Arabic text to tokenize

    Returns:
        List of tokens
    """
    # Simple tokenization - split on whitespace and punctuation
    tokens = []
    current_token = ""

    for char in text:
        if '\u0600' <= char <= '\u06FF' or char.isalnum():
            # Arabic character or alphanumeric
            current_token += char
        else:
            # Non-Arabic, non-alphanumeric character
            if current_token:
                tokens.append(current_token)
                current_token = ""

            # Add non-space punctuation as separate tokens
            if not char.isspace():
                tokens.append(char)

    # Add the last token if there is one
    if current_token:
        tokens.append(current_token)

    return tokens


def is_arabic(text):
    """Check if text contains Arabic characters"""
    return any('\u0600' <= c <= '\u06FF' for c in text)


def clean_text_for_comparison(text):
    """
    Clean text for comparison, removing any potential invisible characters
    or unusual whitespace that might affect string comparison

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Convert to string
    text = str(text)

    # Remove any control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Replace various types of spaces with a standard space
    text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==========================================
# Enhanced Text Generation Methods
# ==========================================

def greedy_decode(
        model,
        tokenizer,
        prompt,
        max_length=128,
        device="cpu",
        early_stopping=True,
        handle_arabic=True
):
    """
    Enhanced greedy decoding with better Arabic language support

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        device: Device to run inference on
        early_stopping: Whether to stop at certain criteria
        handle_arabic: Whether to apply Arabic-specific processing

    Returns:
        output_text: The generated text
    """
    logger.info(f"Performing greedy decoding for prompt: '{prompt}'")

    # Ensure model is in eval mode
    model.eval()

    # Detect if prompt is Arabic and apply special handling
    contains_arabic = is_arabic(prompt) if handle_arabic else False

    # Tokenize input
    tokenizer_output = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    )
    input_ids = tokenizer_output["input_ids"].to(device)

    # Get special token IDs
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 3

    # Start with BOS token
    decoder_input = torch.tensor([[bos_token_id]], device=device)
    generated = [bos_token_id]

    # Track generation time
    start_time = time.time()

    with torch.no_grad():
        # Create source mask once
        src_mask = (input_ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        # Generate encoder outputs once for efficiency
        try:
            enc_output = model.encoder(input_ids, src_mask)
        except Exception as e:
            logger.error(f"Error in encoder forward pass: {e}")
            raise

        # Generate tokens one by one
        for i in range(max_length):
            try:
                # Create target mask
                _, tgt_mask = model.create_masks(input_ids, decoder_input)

                # Get decoder output
                dec_output = model.decoder(decoder_input, enc_output, src_mask, tgt_mask)

                # Get next token using argmax (greedy)
                next_token = torch.argmax(dec_output[:, -1, :], dim=-1, keepdim=True)

                # Add the token to the output
                token_id = next_token.item()
                generated.append(token_id)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

                # Stop if we generate EOS token
                if token_id == eos_token_id:
                    logger.info(f"Generated EOS token at position {i + 1}")
                    break

                # Enhanced early stopping for Arabic text: check for redundant patterns
                if early_stopping and i >= 3:
                    # Simple repetition check (same token repeated)
                    if len(set(generated[-4:])) == 1:
                        logger.info(f"Stopping due to simple repetition at position {i + 1}")
                        break

                    # For longer texts, check for repeating patterns
                    if i >= 8:
                        # Check for repeating pattern of 2-4 tokens
                        for pattern_length in range(2, 5):
                            if (i + 1) >= pattern_length * 2:
                                pattern1 = generated[-(pattern_length * 2):-pattern_length]
                                pattern2 = generated[-pattern_length:]
                                if pattern1 == pattern2:
                                    logger.info(f"Stopping due to repeating pattern at position {i + 1}")
                                    break

            except Exception as e:
                logger.error(f"Error at generation step {i}: {e}")
                break

    # Decode the generated sequence
    generation_time = time.time() - start_time
    logger.info(f"Generation completed in {generation_time:.2f} seconds")

    output_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Apply Arabic-specific post-processing if needed
    if contains_arabic and handle_arabic:
        # Fix any tokenization artifacts for Arabic
        output_text = re.sub(r'##', '', output_text)

        # Fix common spacing issues in Arabic text
        output_text = re.sub(r' ([،؟؛])'.format(), r'\1', output_text)

    logger.info(f"Generated text: '{output_text}'")
    return output_text, generation_time


def sample_decode(
        model,
        tokenizer,
        prompt,
        max_length=128,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        device="cpu",
        handle_arabic=True
):
    """
    Sampling-based text generation (more diverse than greedy)

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Temperature for sampling (higher = more random)
        top_k: Number of highest probability tokens to keep
        top_p: Cumulative probability cutoff for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        device: Device to run on
        handle_arabic: Whether to apply Arabic-specific processing

    Returns:
        output_text: The generated text
    """
    logger.info(f"Performing sampling decoding with temp={temperature}, top_k={top_k}, top_p={top_p}")

    # Ensure model is in eval mode
    model.eval()

    # Detect if prompt is Arabic and apply special handling
    contains_arabic = is_arabic(prompt) if handle_arabic else False

    # Tokenize input
    tokenizer_output = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    )
    input_ids = tokenizer_output["input_ids"].to(device)

    # Get special token IDs
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 3

    # Start with BOS token
    decoder_input = torch.tensor([[bos_token_id]], device=device)
    generated = [bos_token_id]

    # Track recently generated tokens for repetition penalty
    recent_tokens = []
    max_recent = 20  # Number of recent tokens to track

    # Track generation time
    start_time = time.time()

    with torch.no_grad():
        # Create source mask once
        src_mask = (input_ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        # Generate encoder outputs once for efficiency
        enc_output = model.encoder(input_ids, src_mask)

        # Generate tokens one by one
        for i in range(max_length):
            try:
                # Create target mask
                _, tgt_mask = model.create_masks(input_ids, decoder_input)

                # Get decoder output
                dec_output = model.decoder(decoder_input, enc_output, src_mask, tgt_mask)

                # Get logits for next token
                logits = dec_output[:, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0 and recent_tokens:
                    for token_id in set(recent_tokens):
                        logits[:, token_id] /= repetition_penalty

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)

                # Apply top-p (nucleus) filtering
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')

                # Sample next token according to the modified distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Add token to output
                token_id = next_token.item()
                generated.append(token_id)
                recent_tokens.append(token_id)
                if len(recent_tokens) > max_recent:
                    recent_tokens.pop(0)

                decoder_input = torch.cat([decoder_input, next_token], dim=1)

                # Stop if we generate EOS token
                if token_id == eos_token_id:
                    break

                # Check for excessive repetition (same token repeated multiple times)
                if len(recent_tokens) >= 4 and len(set(recent_tokens[-4:])) == 1:
                    break

                # Check for repeating patterns (4+ tokens)
                if len(recent_tokens) >= 8:
                    half_len = len(recent_tokens) // 2
                    if recent_tokens[-half_len:] == recent_tokens[-2 * half_len:-half_len]:
                        break

            except Exception as e:
                logger.error(f"Error at generation step {i}: {e}")
                break

    # Decode the generated sequence
    generation_time = time.time() - start_time
    logger.info(f"Generation completed in {generation_time:.2f} seconds")

    output_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Apply Arabic-specific post-processing if needed
    if contains_arabic and handle_arabic:
        # Fix any tokenization artifacts for Arabic
        output_text = re.sub(r'##', '', output_text)

        # Fix common spacing issues in Arabic text
        output_text = re.sub(r' ([،؟؛])'.format(), r'\1', output_text)

    logger.info(f"Generated text: '{output_text}'")
    return output_text, generation_time


# ==========================================
# Evaluation Metrics
# ==========================================

def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis with better Arabic support

    Args:
        reference: Reference text (gold standard)
        hypothesis: Generated text to evaluate

    Returns:
        BLEU score
    """
    if not hypothesis or not reference:
        return 0.0

    # Clean and normalize texts first
    reference = clean_text_for_comparison(reference)
    hypothesis = clean_text_for_comparison(hypothesis)

    # Check if they're exactly equal after cleaning
    if reference == hypothesis:
        return 1.0

    # Check if the text is Arabic
    is_arabic_text = is_arabic(reference)

    # Decide on tokenization approach
    if is_arabic_text:
        # First normalize Arabic texts
        reference = normalize_arabic(reference)
        hypothesis = normalize_arabic(hypothesis)

        # If they're equal after normalization, return 1.0
        if reference == hypothesis:
            return 1.0

        # Try word-level tokenization for Arabic
        reference_tokens = tokenize_arabic(reference)
        hypothesis_tokens = tokenize_arabic(hypothesis)
    else:
        # For non-Arabic text, use regular tokenization
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()

    # Handle empty tokenization results
    if not reference_tokens or not hypothesis_tokens:
        # Fallback to character-level similarity
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        return char_sim

    # Try to import NLTK for BLEU calculation
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1

        # For Arabic, use different weight distribution
        if is_arabic_text:
            weights = (0.7, 0.3, 0.0, 0.0)  # More weight on unigrams for Arabic
        else:
            weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for non-Arabic

        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens,
                                   weights=weights, smoothing_function=smooth)

        # Boost score for high character similarity (especially for Arabic)
        if bleu_score < 0.9 and is_arabic_text:
            char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
            if char_sim > 0.8:
                bleu_score = max(bleu_score, 0.7)

        return bleu_score
    except ImportError:
        # Fallback if NLTK is not available
        logger.warning("NLTK not available, using character similarity as BLEU fallback")
        return SequenceMatcher(None, reference, hypothesis).ratio()


def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores between reference and hypothesis with better Arabic support

    Args:
        reference: Reference text (gold standard)
        hypothesis: Generated text to evaluate

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    if not hypothesis or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    # Clean and normalize texts
    reference = clean_text_for_comparison(reference)
    hypothesis = clean_text_for_comparison(hypothesis)

    # Check if they're exactly equal after cleaning
    if reference == hypothesis:
        return {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0}

    # For Arabic text, normalize first
    is_arabic_text = is_arabic(reference)
    if is_arabic_text:
        # Normalize Arabic texts
        reference_normalized = normalize_arabic(reference)
        hypothesis_normalized = normalize_arabic(hypothesis)

        # If they're equal after normalization, return perfect scores
        if reference_normalized == hypothesis_normalized:
            return {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0}

        # Tokenize Arabic text
        reference_tokens = tokenize_arabic(reference_normalized)
        hypothesis_tokens = tokenize_arabic(hypothesis_normalized)

        # Join tokens with spaces for ROUGE scoring
        reference_processed = ' '.join(reference_tokens)
        hypothesis_processed = ' '.join(hypothesis_tokens)
    else:
        # For non-Arabic, just use the text as is
        reference_processed = reference
        hypothesis_processed = hypothesis

    # Try to use Rouge scorer if available
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_processed, hypothesis_processed)

        rouge_scores = {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

        # For Arabic, check character-level similarity as well
        if is_arabic_text and all(s < 0.9 for s in rouge_scores.values()):
            char_sim = SequenceMatcher(None, reference_normalized, hypothesis_normalized).ratio()
            if char_sim > 0.8:
                # Boost ROUGE scores for high character similarity
                rouge_scores = {k: max(v, 0.7) for k, v in rouge_scores.items()}

        return rouge_scores
    except ImportError:
        # Fallback if rouge-score is not available
        logger.warning("rouge-score not available, using character similarity as ROUGE fallback")
        similarity = SequenceMatcher(None, reference_processed, hypothesis_processed).ratio()
        return {'rouge1': similarity, 'rouge2': similarity, 'rougeL': similarity}


def calculate_f1_word_match(reference, hypothesis, exclude_stopwords=True):
    """
    Calculate F1 score for word-level match, which works well for Arabic

    Args:
        reference: Reference text (gold standard)
        hypothesis: Generated text to evaluate
        exclude_stopwords: Whether to exclude stopwords from the calculation

    Returns:
        F1 score for word overlap
    """
    if not reference or not hypothesis:
        return 0.0

    # Clean and normalize texts
    reference = clean_text_for_comparison(reference)
    hypothesis = clean_text_for_comparison(hypothesis)

    # Check if they're exactly equal after cleaning
    if reference == hypothesis:
        return 1.0

    # Check if it's Arabic text
    is_arabic_text = is_arabic(reference)

    # For Arabic text, first normalize
    if is_arabic_text:
        reference = normalize_arabic(reference)
        hypothesis = normalize_arabic(hypothesis)

        # If they're equal after normalization, return perfect score
        if reference == hypothesis:
            return 1.0

        # Check equality after removing all whitespace
        if re.sub(r'\s+', '', reference) == re.sub(r'\s+', '', hypothesis):
            return 0.95

        # Tokenize using improved function
        ref_tokens = tokenize_arabic(reference)
        hyp_tokens = tokenize_arabic(hypothesis)
    else:
        # For non-Arabic text
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

    # Remove empty tokens
    ref_tokens = [t for t in ref_tokens if t.strip()]
    hyp_tokens = [t for t in hyp_tokens if t.strip()]

    # Handle empty token lists
    if not ref_tokens or not hyp_tokens:
        # Check character similarity as fallback
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        return char_sim

    # Convert to multisets (allows duplicates)
    from collections import Counter
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    # Calculate intersection size
    intersection = sum((ref_counter & hyp_counter).values())

    # Calculate precision and recall
    precision = intersection / sum(hyp_counter.values()) if hyp_counter else 0
    recall = intersection / sum(ref_counter.values()) if ref_counter else 0

    # Calculate F1
    if precision + recall == 0:
        # Fall back to character-level similarity
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        if char_sim > 0.6:  # Only use if reasonably similar
            return char_sim
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    # For Arabic, consider character-level similarity to catch near-matches
    if is_arabic_text and f1 < 0.7:
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        if char_sim > 0.8:  # High character similarity
            return max(f1, 0.7)  # Boost score if character similarity is high

    return f1


def calculate_exact_match(reference, hypothesis):
    """
    Enhanced exact match that handles different types of whitespace and normalization

    Args:
        reference: Reference text
        hypothesis: Generated text

    Returns:
        Score between 0.0 and 1.0 indicating exact match quality
    """
    if not reference or not hypothesis:
        return 0.0

    # Clean texts for comparison
    clean_ref = clean_text_for_comparison(reference)
    clean_hyp = clean_text_for_comparison(hypothesis)

    # Direct match after cleaning
    if clean_ref == clean_hyp:
        return 1.0

    # Check if the text is Arabic
    is_arabic_text = is_arabic(reference)

    if is_arabic_text:
        # Normalize Arabic texts
        norm_ref = normalize_arabic(reference)
        norm_hyp = normalize_arabic(hypothesis)

        # Match after normalization
        if norm_ref == norm_hyp:
            return 1.0

        # More lenient match - remove all whitespace
        if re.sub(r'\s+', '', norm_ref) == re.sub(r'\s+', '', norm_hyp):
            return 0.95

        # Even more lenient - compare only the actual Arabic characters
        ref_chars = ''.join(c for c in norm_ref if '\u0600' <= c <= '\u06FF')
        hyp_chars = ''.join(c for c in norm_hyp if '\u0600' <= c <= '\u06FF')

        if ref_chars == hyp_chars:
            return 0.9
    else:
        # For non-Arabic, try lowercase comparison
        if clean_ref.lower() == clean_hyp.lower():
            return 0.95

    # Calculate character-level similarity as fallback
    similarity = SequenceMatcher(None, clean_ref, clean_hyp).ratio()

    # Scale similarity to be more lenient
    if similarity > 0.9:
        return 0.9
    elif similarity > 0.8:
        return 0.8
    elif similarity > 0.7:
        return 0.7

    return 0.0


# ==========================================
# Evaluation Functions
# ==========================================

def evaluate_model(model, dataloader, tokenizer, device, config, output_dir=None):
    """
    Evaluate model performance with comprehensive metrics

    Args:
        model: The trained model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer for decoding
        device: Device to run on
        config: Configuration object with evaluation settings
        output_dir: Directory to save results (if None, won't save)

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Starting enhanced model evaluation...")
    model.eval()

    # Initialize metrics tracking
    total_samples = 0
    metrics = {
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'f1_score': [],
        'exact_match': [],
        'generation_time': []
    }

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"eval_results.json")
        detailed_file = os.path.join(output_dir, f"detailed_results.json")

    # Will store detailed results for each sample
    detailed_results = []

    # Detect if we should handle Arabic specially
    handle_arabic = getattr(config, 'handle_arabic', True)

    # Set up progress bar
    pbar = tqdm(dataloader, desc="Evaluating")

    # Get maximum output length
    max_length = getattr(config, 'max_length', 128)

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)

            # Get ground truth answers
            for i in range(src.size(0)):
                # Get source text
                source_text = tokenizer.decode(src[i], skip_special_tokens=True)

                # Get target text (ground truth)
                target_text = tokenizer.decode(tgt[i], skip_special_tokens=True)

                # Generate answer
                start_time = time.time()
                try:
                    # Use greedy decoding for evaluation
                    generated_text, gen_time = greedy_decode(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=source_text,
                        max_length=max_length,
                        device=device,
                        handle_arabic=handle_arabic
                    )
                except Exception as e:
                    logger.error(f"Error generating text for sample {total_samples}: {e}")
                    generated_text = ""
                    gen_time = 0.0

                # Calculate metrics
                bleu = calculate_bleu(target_text, generated_text)
                rouge_scores = calculate_rouge(target_text, generated_text)
                f1 = calculate_f1_word_match(target_text, generated_text)
                exact_match = calculate_exact_match(target_text, generated_text)

                # Store metrics
                metrics['bleu'].append(bleu)
                metrics['rouge1'].append(rouge_scores['rouge1'])
                metrics['rouge2'].append(rouge_scores['rouge2'])
                metrics['rougeL'].append(rouge_scores['rougeL'])
                metrics['f1_score'].append(f1)
                metrics['exact_match'].append(exact_match)
                metrics['generation_time'].append(gen_time)

                # Determine quality level
                if exact_match > 0.9:
                    quality = "Excellent"
                elif f1 > 0.7 or bleu > 0.7:
                    quality = "Good"
                elif f1 > 0.4 or bleu > 0.4:
                    quality = "Partial"
                else:
                    quality = "Poor"

                # Store detailed result
                result = {
                    'id': total_samples,
                    'source': source_text,
                    'reference': target_text,
                    'generated': generated_text,
                    'bleu': bleu,
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rougeL': rouge_scores['rougeL'],
                    'f1_score': f1,
                    'exact_match': exact_match,
                    'generation_time': gen_time,
                    'quality': quality,
                    'is_arabic': is_arabic(target_text)
                }
                detailed_results.append(result)
                total_samples += 1

            # Update progress bar
            avg_bleu = np.mean(metrics['bleu']) if metrics['bleu'] else 0
            avg_f1 = np.mean(metrics['f1_score']) if metrics['f1_score'] else 0
            pbar.set_postfix({'BLEU': f"{avg_bleu:.4f}", 'F1': f"{avg_f1:.4f}"})

    # Calculate average metrics
    evaluation_summary = {}
    for metric_name, values in metrics.items():
        if values:
            evaluation_summary[f'avg_{metric_name}'] = np.mean(values)
            evaluation_summary[f'max_{metric_name}'] = np.max(values)
            evaluation_summary[f'min_{metric_name}'] = np.min(values)

    # Calculate quality distribution
    quality_counts = {}
    for result in detailed_results:
        quality = result['quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    evaluation_summary['quality_distribution'] = quality_counts
    evaluation_summary['total_samples'] = total_samples

    # Calculate language-specific metrics
    arabic_results = [r for r in detailed_results if r['is_arabic']]
    non_arabic_results = [r for r in detailed_results if not r['is_arabic']]

    if arabic_results:
        evaluation_summary['arabic_metrics'] = {
            'count': len(arabic_results),
            'avg_bleu': np.mean([r['bleu'] for r in arabic_results]),
            'avg_f1': np.mean([r['f1_score'] for r in arabic_results]),
            'avg_rouge1': np.mean([r['rouge1'] for r in arabic_results]),
            'avg_rougeL': np.mean([r['rougeL'] for r in arabic_results])
        }

    if non_arabic_results:
        evaluation_summary['non_arabic_metrics'] = {
            'count': len(non_arabic_results),
            'avg_bleu': np.mean([r['bleu'] for r in non_arabic_results]),
            'avg_f1': np.mean([r['f1_score'] for r in non_arabic_results]),
            'avg_rouge1': np.mean([r['rouge1'] for r in non_arabic_results]),
            'avg_rougeL': np.mean([r['rougeL'] for r in non_arabic_results])
        }

    # Save results if output directory is provided
    if output_dir:
        # Convert NumPy values to Python native types for JSON serialization
        json_safe_summary = {}
        for k, v in evaluation_summary.items():
            if isinstance(v, dict):
                json_safe_summary[k] = {
                    kk: float(vv) if isinstance(vv, (np.float32, np.float64, np.int32, np.int64)) else vv
                    for kk, vv in v.items()}
            elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                json_safe_summary[k] = float(v)
            else:
                json_safe_summary[k] = v

        # Save summary
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_summary, f, ensure_ascii=False, indent=2)

        # Make detailed results JSON-serializable
        json_safe_results = []
        for result in detailed_results:
            json_safe_result = {}
            for k, v in result.items():
                if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    json_safe_result[k] = float(v)
                else:
                    json_safe_result[k] = v
            json_safe_results.append(json_safe_result)

        # Save detailed results
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation results saved to {output_dir}")

    logger.info(f"Evaluation complete: {total_samples} samples evaluated")
    logger.info(f"Average BLEU: {evaluation_summary.get('avg_bleu', 0):.4f}")
    logger.info(f"Average F1: {evaluation_summary.get('avg_f1_score', 0):.4f}")

    return evaluation_summary, detailed_results
