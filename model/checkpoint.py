"""Portable training checkpoints, including the tokenizer and model configuration."""

from pathlib import Path

import torch

from .model_conf import ModelConfig


CONFIG_FIELDS = (
    "vocab_size", "block_size", "n_embd", "n_head", "n_layer", "dropout", "batch_size",
)


def config_dict(config):
    return {name: getattr(config, name) for name in CONFIG_FIELDS}


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_config(checkpoint):
    config = ModelConfig()
    for name in CONFIG_FIELDS:
        setattr(config, name, checkpoint["config"][name])
    return config


def save_checkpoint(path, state):
    """Replace a completed checkpoint atomically; retain the old file if writing fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
