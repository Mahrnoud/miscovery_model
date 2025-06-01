"""
Debug and fix script for checkpoint loading issues.
Provides multiple solutions for PyTorch 2.6 weights_only security changes.
"""

import torch
import argparse
import logging
import os
import pickle
from typing import Dict, Any

from utils.greedy_inference import Transformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_checkpoint_contents(checkpoint_path: str) -> None:
    """
    Debug function to inspect checkpoint contents without loading the full model.
    """
    logger.info(f"=== DEBUGGING CHECKPOINT: {checkpoint_path} ===")

    try:
        # Method 1: Try to inspect with pickle directly
        logger.info("Attempting to inspect with pickle...")
        with open(checkpoint_path, 'rb') as f:
            # Just peek at the first few bytes to see the pickle protocol
            first_bytes = f.read(100)
            logger.info(f"First 100 bytes: {first_bytes[:50]}...")

        # Method 2: Try loading with weights_only=False (if you trust the source)
        logger.info("Attempting to load with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Inspect the structure
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint is a dictionary with keys: {list(checkpoint.keys())}")
            for key, value in checkpoint.items():
                logger.info(f"  {key}: {type(value)} - {getattr(value, 'shape', 'no shape')}")
        else:
            logger.info(f"Checkpoint type: {type(checkpoint)}")

        return checkpoint

    except Exception as e:
        logger.error(f"Debug inspection failed: {e}")
        return None


def load_checkpoint_safe(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint with multiple fallback methods.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Method 1: Try with safe globals for numpy
    try:
        logger.info("Attempting Method 1: Using safe_globals for numpy...")
        import numpy as np

        # Add numpy globals to safe list
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.ndarray, np.dtype]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            logger.info("✅ Method 1 succeeded!")
            return checkpoint
    except Exception as e:
        logger.warning(f"Method 1 failed: {e}")

    # Method 2: Use weights_only=False (less secure but works)
    try:
        logger.info("Attempting Method 2: Using weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logger.info("✅ Method 2 succeeded!")
        return checkpoint
    except Exception as e:
        logger.warning(f"Method 2 failed: {e}")

    # Method 3: Try adding more safe globals
    try:
        logger.info("Attempting Method 3: Adding more numpy safe globals...")
        import numpy as np

        safe_globals_list = [
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.core.multiarray._reconstruct,
            np.core.multiarray.array,
            getattr(np, 'int64', None),
            getattr(np, 'float32', None),
            getattr(np, 'float64', None),
        ]
        # Filter out None values
        safe_globals_list = [g for g in safe_globals_list if g is not None]

        with torch.serialization.safe_globals(safe_globals_list):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            logger.info("✅ Method 3 succeeded!")
            return checkpoint
    except Exception as e:
        logger.warning(f"Method 3 failed: {e}")

    # Method 4: Manual safe loading with add_safe_globals
    try:
        logger.info("Attempting Method 4: Using add_safe_globals...")
        import numpy as np

        # Permanently add numpy globals (affects global state)
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype
        ])

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        logger.info("✅ Method 4 succeeded!")
        return checkpoint
    except Exception as e:
        logger.warning(f"Method 4 failed: {e}")

    # If all methods fail
    logger.error("All checkpoint loading methods failed!")
    raise RuntimeError("Unable to load checkpoint with any method")


def clean_checkpoint_for_pytorch(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean checkpoint by converting numpy arrays to torch tensors.
    """
    logger.info("Cleaning checkpoint data...")

    def convert_numpy_to_torch(obj):
        """Recursively convert numpy arrays to torch tensors."""
        import numpy as np

        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, np.number):
            return torch.tensor(obj.item())
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_torch(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_numpy_to_torch(item) for item in obj)
        else:
            return obj

    cleaned_checkpoint = convert_numpy_to_torch(checkpoint)
    logger.info("✅ Checkpoint cleaned successfully!")
    return cleaned_checkpoint


def save_model_with_debug(args):
    """
    Enhanced save function with debugging capabilities.
    """
    # First, debug the checkpoint
    debug_info = debug_checkpoint_contents(args.checkpoint_path)

    # Load checkpoint with safe methods
    checkpoint = load_checkpoint_safe(args.checkpoint_path)

    # Clean the checkpoint if needed
    checkpoint = clean_checkpoint_for_pytorch(checkpoint)

    # Continue with original logic
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        logger.info("Found full checkpoint with metadata")
        model_state_dict = checkpoint["model_state_dict"]
        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", None),
            "metrics": checkpoint.get("metrics", {}),
        }
    else:
        logger.info("Found state dict only")
        model_state_dict = checkpoint
        metadata = {}

    # Get config from args
    config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "max_seq_length": args.max_seq_length,
        "vocab_size": args.vocab_size,
        "dropout": 0.0
    }

    # Create model
    logger.info("Creating model instance")
    model = Transformer(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        vocab_size=config["vocab_size"],
        max_len=config["max_seq_length"],
        pad_idx=args.pad_token_id,
        dropout=config["dropout"]
    )

    # Load state dict
    logger.info("Loading state dict into model")
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    # Save models with safe serialization
    logger.info(f"Saving model state dict to {args.output_dir}/model.pth")
    torch.save(model.state_dict(), f"{args.output_dir}/model.pth", _use_new_zipfile_serialization=True)

    logger.info(f"Saving full model to {args.output_dir}/model_full.pth")
    full_data = {
        "model_state_dict": model.state_dict(),
        "config": config,
        **metadata
    }
    torch.save(full_data, f"{args.output_dir}/model_full.pth", _use_new_zipfile_serialization=True)

    # Create quantized version
    logger.info("Creating quantized model")
    model.eval()

    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        logger.info(f"Saving quantized model to {args.output_dir}/model_quantized.pth")
        torch.save(quantized_model.state_dict(), f"{args.output_dir}/model_quantized.pth",
                  _use_new_zipfile_serialization=True)
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")

    logger.info("Model saving complete!")

    return {
        "model_path": f"{args.output_dir}/model.pth",
        "model_full_path": f"{args.output_dir}/model_full.pth",
        "model_quantized_path": f"{args.output_dir}/model_quantized.pth"
    }


def main():
    parser = argparse.ArgumentParser(description="Debug and save model in PyTorch format")

    # Model checkpoint
    parser.add_argument("--checkpoint_path", type=str,
                        default="/Users/mahmoud/Documents/PythonProjects/model_evaluation/data/checkpoint_epoch1_step2.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/mahmoud/Documents/PythonProjects/model_evaluation/data",
                        help="Output directory")
    parser.add_argument("--debug_only", action="store_true",
                        help="Only debug the checkpoint, don't save models")

    # Model config
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=100000, help="Vocabulary size")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Padding token ID")

    args, unknown = parser.parse_known_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.debug_only:
        # Just debug the checkpoint
        debug_checkpoint_contents(args.checkpoint_path)
    else:
        # Save the model with debugging
        result = save_model_with_debug(args)

        # Print results
        print("\nModel saved successfully!")
        print(f"State dict only: {result['model_path']}")
        print(f"Full model with metadata: {result['model_full_path']}")
        if os.path.exists(result['model_quantized_path']):
            print(f"Quantized model: {result['model_quantized_path']}")


if __name__ == "__main__":
    main()
