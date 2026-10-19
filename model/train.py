import torch
import torch.nn as nn
import os

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from .model_conf import ModelConfig
from .build import CasualSelfAttention

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

torch.set_float32_matmul_precision("high")


def get_training_device():
    if torch.cuda.is_available():
        try:
            # CUDA detection alone does not guarantee this build supports the GPU.
            torch.ones(1, device="cuda").sum().item()
            return torch.device("cuda")
        except RuntimeError as error:
            print(f"CUDA execution failed; using CPU. Check the PyTorch CUDA build: {error}")
    return torch.device("cpu")


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of a multi-head attention layer and a feed-forward network.
    args:
        config: ModelConfig object containing model hyperparameters.
        x: Tensor of shape (batch_size, block_size, n_embd) representing the input to the transformer block.

    outputs:
        x: Tensor of shape (batch_size, block_size, n_embd) after passing through the transformer block.

    How it behaves:
        1. Normalizes the input using LayerNorm.
        2. Applies multi-head attention to the normalized input.
        3. Normalizes the output of the attention layer.
        4. Passes the normalized output through a feed-forward network (MLP) with GELU activation and dropout.

    """
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Multi-head attention layer using pyTorch's optimized math
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)

        # feed-forward network (MLP) with GELU activation and dropout for regularization
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(), # Gaussian Error Linear Unit function AKA boring math
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        # pyTorch dot product attention for speed
        # automatically triggers FlashAttention
        norm_x = self.ln_1(x)
        x = x + self.attn(norm_x)

        # Self-Attention pyTorch logic (layers)
        x = x + self.mlp(self.ln_2(x))
        return x


def train_local_model(model, dataset, config, epochs=3):
    """
    Trains the LocalGPT model on the provided dataset.

    Args:
        model: The LocalGPT model to be trained.
        dataset: The TokenizerDataset containing the training data.
        config: ModelConfig object containing model hyperparameters.
        epochs: Number of training epochs (default is 3).
    """

    device = get_training_device()
    model = model.to(device)

    # Gradient Checkpointing
    # Reason: Instead of storing every single mathmatical activation value accross
    # the entire model, we only store the inputs and recompute the activations during
    # the backward pass.
    model.gradient_checkpointing_enable()

    # AdamW Optimizer with 8-bit precision using bitsandbytes library
    if device.type == "cuda":
        try:
            import bitsandbytes as bnb
        except ImportError:
            bnb = None
    else:
        bnb = None
    optimizer_class = bnb.optim.AdamW8bit if bnb is not None else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=6e-4, weight_decay=0.1)

    scaler = GradScaler("cuda", enabled=device.type == "cuda")  # For mixed precision training

    dataLoader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    model.train()  # Set the model to training mode
    print("Starting training...")

    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataLoader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Clear previous gradients

            with autocast(device_type=device.type, enabled=device.type == "cuda"):  # Mixed precision context
                logits, loss = model(x, y)  # Forward pass and loss computation

            scaler.scale(loss).backward()  # Backpropagation with scaled loss

            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)  # Update parameters
            scaler.update()  # Update the scale for next iteration

            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss Value: {loss.item():.4f}")
