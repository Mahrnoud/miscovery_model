#!/usr/bin/env python3
"""
Checkpoint Manager Utility

This script helps you manage model checkpoints and convert them to be compatible
with different PyTorch versions.

Usage:
    python checkpoint_manager.py check <checkpoint_path>
    python checkpoint_manager.py convert <old_path> <new_path>
    python checkpoint_manager.py info <checkpoint_path>
"""

import argparse
import torch
import os
import sys
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_checkpoint(checkpoint_path):
    """Check if a checkpoint can be loaded and provide detailed information."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file '{checkpoint_path}' does not exist")
        return False

    print(f"🔍 Checking checkpoint: {checkpoint_path}")
    print(f"📏 File size: {os.path.getsize(checkpoint_path) / (1024 * 1024):.2f} MB")

    # Check PyTorch version
    print(f"🐍 Current PyTorch version: {torch.__version__}")

    results = {
        'secure_load': False,
        'compat_load': False,
        'has_config': False,
        'has_state_dict': False,
        'keys_count': 0,
        'pytorch_version': None
    }

    # Try secure loading (PyTorch 2.6+ default)
    print("\n🔒 Testing secure loading (weights_only=True)...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        results['secure_load'] = True
        print("  ✅ Success: Checkpoint can be loaded securely")

        if isinstance(checkpoint, dict):
            results['has_config'] = 'config' in checkpoint
            results['has_state_dict'] = 'model_state_dict' in checkpoint
            results['keys_count'] = len(checkpoint.keys()) if isinstance(checkpoint, dict) else 0

            if 'pytorch_version' in checkpoint:
                results['pytorch_version'] = checkpoint['pytorch_version']

            print(f"  📋 Keys in checkpoint: {list(checkpoint.keys())}")

    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:100]}...")

    # Try compatibility loading
    print("\n🔓 Testing compatibility loading (weights_only=False)...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        results['compat_load'] = True
        print("  ✅ Success: Checkpoint can be loaded in compatibility mode")

        if isinstance(checkpoint, dict):
            results['has_config'] = 'config' in checkpoint
            results['has_state_dict'] = 'model_state_dict' in checkpoint
            results['keys_count'] = len(checkpoint.keys()) if isinstance(checkpoint, dict) else 0

            if 'pytorch_version' in checkpoint:
                results['pytorch_version'] = checkpoint['pytorch_version']

            if not results['secure_load']:  # Only print if we didn't already print this
                print(f"  📋 Keys in checkpoint: {list(checkpoint.keys())}")

    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:100]}...")

    # Summary
    print("\n📊 Summary:")
    print(f"  Secure loading (recommended): {'✅' if results['secure_load'] else '❌'}")
    print(f"  Compatibility loading: {'✅' if results['compat_load'] else '❌'}")
    print(f"  Has config: {'✅' if results['has_config'] else '❌'}")
    print(f"  Has model state dict: {'✅' if results['has_state_dict'] else '❌'}")
    print(f"  Number of top-level keys: {results['keys_count']}")

    if results['pytorch_version']:
        print(f"  Saved with PyTorch version: {results['pytorch_version']}")

    # Recommendations
    print("\n💡 Recommendations:")
    if results['secure_load']:
        print("  ✅ This checkpoint is fully compatible with PyTorch 2.6+")
    elif results['compat_load']:
        print("  ⚠️  This checkpoint requires compatibility mode (weights_only=False)")
        print("  💡 Consider converting it to secure format for better security")
    else:
        print("  ❌ This checkpoint cannot be loaded - it may be corrupted")

    return results['secure_load'] or results['compat_load']


def convert_checkpoint(old_path, new_path, force=False):
    """Convert a checkpoint to secure format."""
    if not os.path.exists(old_path):
        print(f"❌ Error: Source checkpoint '{old_path}' does not exist")
        return False

    if os.path.exists(new_path) and not force:
        print(f"❌ Error: Target file '{new_path}' already exists. Use --force to overwrite")
        return False

    print(f"🔄 Converting checkpoint...")
    print(f"  Source: {old_path}")
    print(f"  Target: {new_path}")

    try:
        # Load the old checkpoint
        print("📥 Loading source checkpoint...")
        checkpoint = torch.load(old_path, map_location='cpu', weights_only=False)

        # Create secure version
        print("🔧 Creating secure version...")
        new_checkpoint = {}

        if isinstance(checkpoint, dict):
            # Copy essential keys only
            essential_keys = ['model_state_dict', 'config', 'epoch', 'optimizer_state_dict']

            for key in essential_keys:
                if key in checkpoint:
                    new_checkpoint[key] = checkpoint[key]
                    print(f"  ✅ Copied: {key}")

            # Add any other tensor-based keys that are safe
            for key, value in checkpoint.items():
                if key not in essential_keys and isinstance(value, torch.Tensor):
                    new_checkpoint[key] = value
                    print(f"  ✅ Copied tensor: {key}")
        else:
            # If it's just a state dict
            new_checkpoint = checkpoint

        # Add metadata
        new_checkpoint['pytorch_version'] = torch.__version__
        new_checkpoint['conversion_timestamp'] = time.time()
        new_checkpoint['original_format'] = 'legacy_converted'

        # Ensure directory exists
        os.makedirs(os.path.dirname(new_path) if os.path.dirname(new_path) else '.', exist_ok=True)

        # Save securely
        print("💾 Saving secure checkpoint...")
        torch.save(new_checkpoint, new_path, pickle_protocol=4)

        # Verify conversion
        print("🔍 Verifying conversion...")
        verification_results = check_checkpoint(new_path)

        if verification_results:
            print("✅ Conversion successful!")

            # Show file size comparison
            old_size = os.path.getsize(old_path) / (1024 * 1024)
            new_size = os.path.getsize(new_path) / (1024 * 1024)
            print(f"📏 Size comparison: {old_size:.2f} MB → {new_size:.2f} MB")

            return True
        else:
            print("❌ Conversion failed verification")
            return False

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def get_checkpoint_info(checkpoint_path):
    """Get detailed information about a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file '{checkpoint_path}' does not exist")
        return

    print(f"📋 Detailed checkpoint information: {checkpoint_path}")
    print(f"📁 File path: {os.path.abspath(checkpoint_path)}")
    print(f"📏 File size: {os.path.getsize(checkpoint_path) / (1024 * 1024):.2f} MB")
    print(f"📅 Modified: {time.ctime(os.path.getmtime(checkpoint_path))}")

    try:
        # Try to load with compatibility mode to get full info
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            print(f"\n📊 Checkpoint structure:")
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor {list(value.shape)} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  {key}: Dict with {len(value)} items")
                    if key == 'config':
                        for config_key, config_value in value.items():
                            print(f"    {config_key}: {config_value}")
                elif isinstance(value, (int, float, str)):
                    print(f"  {key}: {type(value).__name__} = {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print(f"\n📊 Checkpoint is a {type(checkpoint).__name__}")
            if hasattr(checkpoint, 'keys'):
                print(f"  Keys: {len(checkpoint.keys())}")
                for i, key in enumerate(list(checkpoint.keys())[:10]):  # Show first 10 keys
                    value = checkpoint[key]
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: Tensor {list(value.shape)}")
                if len(checkpoint.keys()) > 10:
                    print(f"    ... and {len(checkpoint.keys()) - 10} more keys")

    except Exception as e:
        print(f"❌ Could not load checkpoint for detailed analysis: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint Manager - Handle PyTorch model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if a checkpoint can be loaded
  python checkpoint_manager.py check model.pth

  # Convert a checkpoint to secure format
  python checkpoint_manager.py convert old_model.pth new_model.pth

  # Get detailed information about a checkpoint
  python checkpoint_manager.py info model.pth
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check if a checkpoint can be loaded')
    check_parser.add_argument('checkpoint', help='Path to checkpoint file')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert checkpoint to secure format')
    convert_parser.add_argument('source', help='Source checkpoint path')
    convert_parser.add_argument('target', help='Target checkpoint path')
    convert_parser.add_argument('--force', action='store_true', help='Overwrite target if it exists')

    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed checkpoint information')
    info_parser.add_argument('checkpoint', help='Path to checkpoint file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    print(f"🚀 Checkpoint Manager - PyTorch {torch.__version__}")
    print("=" * 60)

    if args.command == 'check':
        success = check_checkpoint(args.checkpoint)
        sys.exit(0 if success else 1)

    elif args.command == 'convert':
        success = convert_checkpoint(args.source, args.target, args.force)
        sys.exit(0 if success else 1)

    elif args.command == 'info':
        get_checkpoint_info(args.checkpoint)


if __name__ == "__main__":
    main()
