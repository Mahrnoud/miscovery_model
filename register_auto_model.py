"""
Register the Miscovery model with Hugging Face Auto classes.
This will allow you to use AutoModel, AutoModelForSeq2SeqLM, etc. with your model.
"""

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, MODEL_MAPPING
from model_architecture import CustomTransformerConfig, CustomTransformerModel


def register_with_auto_classes():
    """Register the Miscovery model with Hugging Face Auto classes."""

    # Register the config with the config mapping
    CONFIG_MAPPING.register("miscovery", CustomTransformerConfig)

    # Register the model with the model mapping
    MODEL_MAPPING.register(CustomTransformerConfig, CustomTransformerModel)

    # Register for sequence-to-sequence tasks
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.register(CustomTransformerConfig, CustomTransformerModel)

    print("Successfully registered the Miscovery model with Hugging Face Auto classes!")
    print("You can now use AutoModel.from_pretrained() and related functions with your model.")


if __name__ == "__main__":
    register_with_auto_classes()