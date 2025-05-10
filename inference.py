"""
Inference script for using the fine-tuned models for:
1. Text Summarization
2. Translation
3. Question-Answering
"""

import argparse
import logging
import random
import sys
import time

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Import our model architecture
from model_architecture import Transformer
from enhanced_evaluation_code import generate_text_optimized

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path, config, device="cuda"):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {model_path}")

    # Create model from config
    model = Transformer(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        vocab_size=config["vocab_size"],
        max_len=config["max_seq_length"],
        pad_idx=config["pad_idx"],
        dropout=0.0  # Use 0 dropout for inference
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def prepare_input_for_task(text, task_type, tokenizer, max_length=512, src_lang="en", tgt_lang="ar"):
    """Prepare input text based on task type"""

    if task_type == "summarization":
        # Add prefix for summarization
        prefix = "summarize: "
        input_text = prefix + text

    elif task_type == "translation":
        # Add language tag
        if src_lang == "en" and tgt_lang == "ar":
            input_text = f"[LANG_EN] {text}"
        elif src_lang == "ar" and tgt_lang == "en":
            input_text = f"[LANG_AR] {text}"
        else:
            raise ValueError(f"Unsupported language pair: {src_lang}-{tgt_lang}")

    elif task_type == "qa":
        # Question should already have context
        if "context:" not in text.lower() and "context :" not in text.lower():
            input_text = f"question: {text}"
            logger.warning("QA input doesn't contain context. Results may be unreliable.")
        else:
            input_text = text

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return inputs["input_ids"]


def run_inference(model, input_text, tokenizer, task_type, args, device="cuda"):
    """Run inference on the input text"""

    # Prepare input
    input_ids = prepare_input_for_task(
        input_text,
        task_type,
        tokenizer,
        max_length=args.max_seq_length,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    input_ids = input_ids.to(device)

    # Generate output
    with torch.no_grad():
        start_time = time.time()

        output = generate_text_optimized(
            model,
            input_ids,
            tokenizer,
            max_length=args.max_output_length,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample
        )

        end_time = time.time()

    # Post-process output
    if task_type == "translation" and output.startswith(f"[LANG_{args.tgt_lang.upper()}]"):
        # Remove language tag
        output = output[len(f"[LANG_{args.tgt_lang.upper()}]"):].strip()

    inference_time = end_time - start_time

    return output, inference_time


def batch_inference(model, input_texts, tokenizer, task_type, args, device="cuda"):
    """Run inference on a batch of input texts"""

    results = []
    total_time = 0

    for text in tqdm(input_texts, desc="Processing"):
        output, inference_time = run_inference(
            model, text, tokenizer, task_type, args, device
        )
        results.append({
            "input": text,
            "output": output,
            "time": inference_time
        })
        total_time += inference_time

    avg_time = total_time / len(input_texts) if input_texts else 0
    logger.info(f"Average inference time: {avg_time:.4f}s per example")

    return results


def save_results(results, output_file):
    """Save results to file"""
    import json

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")

    # Model configuration
    config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "vocab_size": len(tokenizer),
        "max_seq_length": args.max_seq_length,
        "pad_idx": tokenizer.pad_token_id
    }

    # Load model
    model = load_model(args.model_path, config, device)

    # Read input texts
    input_texts = []

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(input_texts)} inputs from {args.input_file}")

    elif args.input_text:
        input_texts = [args.input_text]

    else:
        logger.error("No input provided. Use --input_file or --input_text")
        return

    # Run inference
    results = batch_inference(
        model, input_texts, tokenizer, args.task_type, args, device
    )

    # Save results if output file is specified
    if args.output_file:
        save_results(results, args.output_file)

    # Print results for single input
    if len(input_texts) == 1:
        print("\nInput:", input_texts[0])
        print("\nOutput:", results[0]["output"])
        print(f"\nInference time: {results[0]['time']:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned models")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=12, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum input sequence length")
    parser.add_argument("--max_output_length", type=int, default=256, help="Maximum output generation length")

    # Inference parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling instead of greedy decoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")

    # Task parameters
    parser.add_argument(
        "--task_type", type=str, default="qa",
        choices=["summarization", "translation", "qa"],
        help="Type of task"
    )
    parser.add_argument("--src_lang", type=str, default="en", help="Source language for translation")
    parser.add_argument("--tgt_lang", type=str, default="ar", help="Target language for translation")

    # Input/output parameters
    parser.add_argument("--input_text", default="what is the capital of Egypt?", type=str, help="Single input text")
    parser.add_argument("--input_file", type=str, help="File with input texts (one per line)")
    parser.add_argument("--output_file", type=str, help="Output file to save results")

    # Model loading parameters
    parser.add_argument("--model_path", type=str, default="/Users/mahmoud/Documents/PythonProjects/miscovery_llm/models/checkpoints/model.pth", help="Path to fine-tuned model checkpoint")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer", help="Tokenizer name or path")

    args = parser.parse_args()

    # Check for required args
    if not args.input_text and not args.input_file:
        parser.error("Either --input_text or --input_file must be provided")

    main(args)
