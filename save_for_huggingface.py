"""
Updated script to convert your trained model to Hugging Face format without tensor sharing issues.
"""

import argparse
import os
import logging
import torch
from transformers import AutoTokenizer
from model_architecture import CustomTransformerConfig, CustomTransformerModel, Transformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_for_huggingface(args):
    # Step 1: Load the checkpoint
    logger.info(f"Loading checkpoint from {args.model_path}")

    # Try to load the checkpoint - handle different formats
    try:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        # Check if this is a full checkpoint dict or just the state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            logger.info("Loaded full checkpoint with metadata")
            model_state_dict = checkpoint["model_state_dict"]
        else:
            logger.info("Loaded state dict directly")
            model_state_dict = checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 3: Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Step 4: Create config
    config = CustomTransformerConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        max_position_embeddings=args.max_seq_length,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id,
    )

    # Save the config
    config.save_pretrained(args.output_dir)
    logger.info(f"Saved config to {args.output_dir}")

    # Step 5: First create a fresh instance of your original model architecture
    original_model = Transformer(
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

    # Load the state dict into this model
    missing_keys, unexpected_keys = original_model.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading original model: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading original model: {unexpected_keys}")

    # Step 6: Create a fresh instance of the HF model
    hf_model = CustomTransformerModel(config)

    # Step 7: Key mapping approach 1 - Manual parameter transfer without shared tensors
    # Instead of directly manipulating the state dict, copy parameters tensor by tensor
    logger.info("Transferring parameters from original model to HF model...")

    # First, extract parameters from the original model
    with torch.no_grad():
        # Create a new state dict for the HF model
        new_state_dict = {}

        # Manually map encoder parameters
        # Embedding layer
        new_state_dict["model.encoder.embedding.weight"] = original_model.encoder.embedding.weight.clone()

        # Encoder layers
        for i in range(args.num_encoder_layers):
            # Normalization layers
            new_state_dict[f"model.encoder.layers.{i}.norm1.weight"] = original_model.encoder.layers[
                i].norm1.weight.clone()
            new_state_dict[f"model.encoder.layers.{i}.norm2.weight"] = original_model.encoder.layers[
                i].norm2.weight.clone()

            # Self-attention layers
            layer_prefix = f"model.encoder.layers.{i}.self_attention"
            orig_layer = original_model.encoder.layers[i].self_attention

            new_state_dict[f"{layer_prefix}.q_proj.weight"] = orig_layer.q_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.q_proj.bias"] = orig_layer.q_proj.bias.clone()
            new_state_dict[f"{layer_prefix}.k_proj.weight"] = orig_layer.k_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.k_proj.bias"] = orig_layer.k_proj.bias.clone()
            new_state_dict[f"{layer_prefix}.v_proj.weight"] = orig_layer.v_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.v_proj.bias"] = orig_layer.v_proj.bias.clone()
            new_state_dict[f"{layer_prefix}.out_proj.weight"] = orig_layer.out_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.out_proj.bias"] = orig_layer.out_proj.bias.clone()

            # Layer scale and rotary embeddings if present
            if hasattr(orig_layer, 'layer_scale'):
                new_state_dict[f"{layer_prefix}.layer_scale.gamma"] = orig_layer.layer_scale.gamma.clone()

            if hasattr(orig_layer, 'rotary_emb') and hasattr(orig_layer.rotary_emb, 'inv_freq'):
                new_state_dict[f"{layer_prefix}.rotary_emb.inv_freq"] = orig_layer.rotary_emb.inv_freq.clone()

            # Feed-forward layers
            ff_prefix = f"model.encoder.layers.{i}.feed_forward"
            orig_ff = original_model.encoder.layers[i].feed_forward

            new_state_dict[f"{ff_prefix}.w1.weight"] = orig_ff.w1.weight.clone()
            new_state_dict[f"{ff_prefix}.w1.bias"] = orig_ff.w1.bias.clone() if hasattr(orig_ff.w1,
                                                                                        'bias') and orig_ff.w1.bias is not None else torch.zeros(
                orig_ff.w1.weight.size(0))
            new_state_dict[f"{ff_prefix}.w2.weight"] = orig_ff.w2.weight.clone()
            new_state_dict[f"{ff_prefix}.w2.bias"] = orig_ff.w2.bias.clone() if hasattr(orig_ff.w2,
                                                                                        'bias') and orig_ff.w2.bias is not None else torch.zeros(
                orig_ff.w2.weight.size(0))
            new_state_dict[f"{ff_prefix}.w3.weight"] = orig_ff.w3.weight.clone()
            new_state_dict[f"{ff_prefix}.w3.bias"] = orig_ff.w3.bias.clone() if hasattr(orig_ff.w3,
                                                                                        'bias') and orig_ff.w3.bias is not None else torch.zeros(
                orig_ff.w3.weight.size(0))

            if hasattr(orig_ff, 'layer_scale'):
                new_state_dict[f"{ff_prefix}.layer_scale.gamma"] = orig_ff.layer_scale.gamma.clone()

        # Final encoder norm
        new_state_dict["model.encoder.norm.weight"] = original_model.encoder.norm.weight.clone()

        # Decoder embedding
        new_state_dict["model.decoder.embedding.weight"] = original_model.decoder.embedding.weight.clone()

        # Decoder layers
        for i in range(args.num_decoder_layers):
            # Normalization layers
            new_state_dict[f"model.decoder.layers.{i}.norm1.weight"] = original_model.decoder.layers[
                i].norm1.weight.clone()
            new_state_dict[f"model.decoder.layers.{i}.norm2.weight"] = original_model.decoder.layers[
                i].norm2.weight.clone()
            new_state_dict[f"model.decoder.layers.{i}.norm3.weight"] = original_model.decoder.layers[
                i].norm3.weight.clone()

            # Self-attention layers
            layer_prefix = f"model.decoder.layers.{i}.self_attention"
            orig_layer = original_model.decoder.layers[i].self_attention

            new_state_dict[f"{layer_prefix}.q_proj.weight"] = orig_layer.q_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.q_proj.bias"] = orig_layer.q_proj.bias.clone()
            new_state_dict[f"{layer_prefix}.k_proj.weight"] = orig_layer.k_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.k_proj.bias"] = orig_layer.k_proj.bias.clone()
            new_state_dict[f"{layer_prefix}.v_proj.weight"] = orig_layer.v_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.v_proj.bias"] = orig_layer.v_proj.bias.clone()
            new_state_dict[f"{layer_prefix}.out_proj.weight"] = orig_layer.out_proj.weight.clone()
            new_state_dict[f"{layer_prefix}.out_proj.bias"] = orig_layer.out_proj.bias.clone()

            # Layer scale and rotary embeddings if present
            if hasattr(orig_layer, 'layer_scale'):
                new_state_dict[f"{layer_prefix}.layer_scale.gamma"] = orig_layer.layer_scale.gamma.clone()

            if hasattr(orig_layer, 'rotary_emb') and hasattr(orig_layer.rotary_emb, 'inv_freq'):
                new_state_dict[f"{layer_prefix}.rotary_emb.inv_freq"] = orig_layer.rotary_emb.inv_freq.clone()

            # Cross-attention layers
            cross_prefix = f"model.decoder.layers.{i}.cross_attention"
            orig_cross = original_model.decoder.layers[i].cross_attention

            new_state_dict[f"{cross_prefix}.q_proj.weight"] = orig_cross.q_proj.weight.clone()
            new_state_dict[f"{cross_prefix}.q_proj.bias"] = orig_cross.q_proj.bias.clone()
            new_state_dict[f"{cross_prefix}.k_proj.weight"] = orig_cross.k_proj.weight.clone()
            new_state_dict[f"{cross_prefix}.k_proj.bias"] = orig_cross.k_proj.bias.clone()
            new_state_dict[f"{cross_prefix}.v_proj.weight"] = orig_cross.v_proj.weight.clone()
            new_state_dict[f"{cross_prefix}.v_proj.bias"] = orig_cross.v_proj.bias.clone()
            new_state_dict[f"{cross_prefix}.out_proj.weight"] = orig_cross.out_proj.weight.clone()
            new_state_dict[f"{cross_prefix}.out_proj.bias"] = orig_cross.out_proj.bias.clone()

            if hasattr(orig_cross, 'layer_scale'):
                new_state_dict[f"{cross_prefix}.layer_scale.gamma"] = orig_cross.layer_scale.gamma.clone()

            if hasattr(orig_cross, 'rotary_emb') and hasattr(orig_cross.rotary_emb, 'inv_freq'):
                new_state_dict[f"{cross_prefix}.rotary_emb.inv_freq"] = orig_cross.rotary_emb.inv_freq.clone()

            # Feed-forward layers
            ff_prefix = f"model.decoder.layers.{i}.feed_forward"
            orig_ff = original_model.decoder.layers[i].feed_forward

            new_state_dict[f"{ff_prefix}.w1.weight"] = orig_ff.w1.weight.clone()
            new_state_dict[f"{ff_prefix}.w1.bias"] = orig_ff.w1.bias.clone() if hasattr(orig_ff.w1,
                                                                                        'bias') and orig_ff.w1.bias is not None else torch.zeros(
                orig_ff.w1.weight.size(0))
            new_state_dict[f"{ff_prefix}.w2.weight"] = orig_ff.w2.weight.clone()
            new_state_dict[f"{ff_prefix}.w2.bias"] = orig_ff.w2.bias.clone() if hasattr(orig_ff.w2,
                                                                                        'bias') and orig_ff.w2.bias is not None else torch.zeros(
                orig_ff.w2.weight.size(0))
            new_state_dict[f"{ff_prefix}.w3.weight"] = orig_ff.w3.weight.clone()
            new_state_dict[f"{ff_prefix}.w3.bias"] = orig_ff.w3.bias.clone() if hasattr(orig_ff.w3,
                                                                                        'bias') and orig_ff.w3.bias is not None else torch.zeros(
                orig_ff.w3.weight.size(0))

            if hasattr(orig_ff, 'layer_scale'):
                new_state_dict[f"{ff_prefix}.layer_scale.gamma"] = orig_ff.layer_scale.gamma.clone()

        # Final decoder norm and output projection
        new_state_dict["model.decoder.norm.weight"] = original_model.decoder.norm.weight.clone()
        new_state_dict[
            "model.decoder.output_projection.weight"] = original_model.decoder.output_projection.weight.clone()
        new_state_dict["model.decoder.output_projection.bias"] = original_model.decoder.output_projection.bias.clone()

    # Step 8: Load the new state dict into the HF model
    logger.info("Loading parameters into HF model...")
    try:
        # Set strict=False to allow for missing keys or unexpected keys
        missing_keys, unexpected_keys = hf_model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        raise

    # Step 9: Save the HF model with safe_serialization=False to avoid tensor sharing issues
    logger.info("Saving HF model...")
    try:
        hf_model.save_pretrained(args.output_dir, safe_serialization=False)
        logger.info(f"Successfully saved model to {args.output_dir}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

    # Step 10: Save the tokenizer
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved tokenizer to {args.output_dir}")

    # Step 11: Create model card
    model_card = f"""---
language: multilingual
license: mit
tags:
  - transformer
  - summarization
  - translation
  - question-answering
  - english
  - arabic
---

# Miscovery Transformer Model

This model is a transformer-based encoder-decoder model for multiple NLP tasks:
- Text summarization
- Translation (English-Arabic)
- Question-answering

## Model Architecture

- Model type: miscovery
- Number of parameters: {sum(p.numel() for p in hf_model.parameters())}
- Encoder layers: {args.num_encoder_layers}
- Decoder layers: {args.num_decoder_layers}
- Attention heads: {args.num_heads}
- Hidden size: {args.d_model}
- Feed-forward size: {args.d_ff}

## Training

The model was trained in two stages:
1. Pre-training on sentence rearrangement tasks
2. Fine-tuning on downstream tasks

## Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{args.model_name}")
model = AutoModel.from_pretrained("{args.model_name}")

# For summarization
input_text = "summarize: " + text_to_summarize
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=150)
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# For translation
input_text = "[LANG_EN] " + text_to_translate
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=150)
translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

## Limitations

This model was trained on specific datasets and may not generalize well to all domains.
"""

    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(model_card)
    logger.info(f"Created model card at {os.path.join(args.output_dir, 'README.md')}")

    # Step 12: Create any necessary auto registration files
    with open(os.path.join(args.output_dir, "__init__.py"), "w") as f:
        f.write('''"""Miscovery model package."""

from .configuration_miscovery import CustomTransformerConfig
from .modeling_miscovery import CustomTransformerModel

__all__ = [
    "CustomTransformerConfig",
    "CustomTransformerModel",
]''')
    logger.info("Created __init__.py to register the model")

    # Copy the model architecture files with proper names
    logger.info("Creating necessary model architecture files...")

    # Create configuration file
    with open(os.path.join(args.output_dir, "configuration_miscovery.py"), "w") as f:
        f.write('''"""Miscovery model configuration"""

from transformers.configuration_utils import PretrainedConfig

class CustomTransformerConfig(PretrainedConfig):
    model_type = "miscovery"

    def __init__(
            self,
            vocab_size=100000,
            d_model=768,
            num_heads=12,
            d_ff=3072,
            num_encoder_layers=12,
            num_decoder_layers=12,
            max_position_embeddings=512,
            dropout=0.1,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_flash_attn=False,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
''')

    # Create modeling file (simplified version that imports the real implementation)
    with open(os.path.join(args.output_dir, "modeling_miscovery.py"), "w") as f:
        f.write('''"""Miscovery model implementation"""

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn

from .configuration_miscovery import CustomTransformerConfig

# Import the actual model architecture
# This is a simplified placeholder that should be replaced with your actual model code
class CustomTransformerModel(PreTrainedModel):
    config_class = CustomTransformerConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        # Initialize model components
        # This will need to be replaced with your actual model architecture
        self.model = None  # Your model implementation here

    def forward(
            self,
            input_ids=None,
            decoder_input_ids=None,
            attention_mask=None,
            decoder_attention_mask=None,
            labels=None,
            **kwargs
    ):
        # Forward pass implementation
        # This will need to be replaced with your actual forward method
        return Seq2SeqLMOutput(
            loss=None,
            logits=None,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # Prepare inputs implementation for generation
        # This will need to be replaced with your actual method
        pass
''')

    # Step 13: Log success message
    logger.info(f"""
Success! Model saved to {args.output_dir} with safe_serialization=False

To upload to Hugging Face Hub:
1. Create a repository on huggingface.co/new
2. Run the following commands:
    git init
    git remote add origin https://huggingface.co/{args.model_name}
    git add .
    git commit -m "Initial commit"
    git push -u origin main

Or use the Hugging Face CLI:
    huggingface-cli login
    huggingface-cli upload {args.output_dir} {args.model_name}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to Hugging Face format")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=12, help="Number of decoder layers")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (set to 0 for inference)")

    # I/O parameters
    parser.add_argument("--model_path", type=str,
                        default="stage_02/output/checkpoints/best_model.pth",
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--output_dir", type=str, default="/content/finetune_output/model",
                        help="Directory to save the HF model")
    parser.add_argument("--tokenizer_name", type=str, default="miscovery/tokenizer", help="Tokenizer name or path")
    parser.add_argument("--model_name", type=str, default="miscovery/model",
                        help="Name for the model on Hugging Face Hub")

    args, unknown = parser.parse_known_args()

    save_for_huggingface(args)
