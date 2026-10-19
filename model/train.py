import torch
import torch.nn as nn
import os
import hashlib
from pathlib import Path

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from .model_conf import ModelConfig
from .build import CasualSelfAttention
from .checkpoint import config_dict, save_checkpoint
from .evaluate import evaluate_model

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


def create_optimizer(model, device, optimizer_type=None):
    """Honor a checkpoint's optimizer before considering optional dependencies."""
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    if optimizer_type not in {None, "AdamW8bit"}:
        raise ValueError(f"Unsupported checkpoint optimizer: {optimizer_type}")
    if optimizer_type == "AdamW8bit" and device.type != "cuda":
        raise ValueError("This AdamW8bit checkpoint requires working CUDA to resume; use --evaluate-only for inference")
    if device.type == "cuda":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            if optimizer_type == "AdamW8bit":
                raise ValueError(
                    "This checkpoint requires bitsandbytes. Install it with "
                    "local_ai_env/Scripts/python.exe -m pip install bitsandbytes, then resume again"
                ) from error
            bnb = None
    else:
        bnb = None
    # bitsandbytes publicly re-exports AdamW8bit without an explicit typed re-export.
    optimizer_class = bnb.optim.AdamW8bit if bnb is not None else torch.optim.AdamW  # pyright: ignore[reportPrivateImportUsage]
    return optimizer_class(model.parameters(), lr=6e-4, weight_decay=0.1)


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


def train_local_model(model, dataset, config, epochs=3, *, checkpoint_dir=None,
                      resume=None, tokenizer_json=None, validation_dataset=None,
                      save_every=100):
    """
    Trains the LocalGPT model on the provided dataset.

    Args:
        model: The LocalGPT model to be trained.
        dataset: The TokenizerDataset containing the training data.
        config: ModelConfig object containing model hyperparameters.
        epochs: Number of training epochs (default is 3).
        checkpoint_dir: Folder for latest.pt and best.pt; None disables saving.
        resume: Loaded checkpoint containing training state, if continuing a run.
        tokenizer_json: Tokenizer serialized into the checkpoint for later inference.
        validation_dataset: Held-out contexts used to compare epochs.
        save_every: Save latest.pt after this many batches and at every epoch end.
    """

    if epochs < 1 or save_every < 1:
        raise ValueError("epochs and save_every must be positive")
    device = get_training_device()
    model = model.to(device)

    # Gradient Checkpointing
    # Reason: Instead of storing every single mathmatical activation value accross
    # the entire model, we only store the inputs and recompute the activations during
    # the backward pass.
    model.gradient_checkpointing_enable()

    # AdamW Optimizer with 8-bit precision using bitsandbytes library
    optimizer_type = resume["optimizer_type"] if resume is not None else None
    optimizer = create_optimizer(model, device, optimizer_type)
    print(f"Training device: {device} | Optimizer: {type(optimizer).__name__}")

    scaler = GradScaler("cuda", enabled=device.type == "cuda")  # For mixed precision training

    signature = hashlib.sha256(dataset.data.numpy().tobytes()).hexdigest()
    data_info = {"sha256": signature, "block_size": dataset.block_size, "stride": dataset.stride}
    start_epoch, next_batch, global_step = 0, 0, 0
    best_validation_loss = float("inf")
    if resume is not None:
        if resume["config"] != config_dict(config) or resume["data_info"] != data_info:
            raise ValueError("Checkpoint configuration or training data differs from this run")
        if resume["tokenizer_json"] != tokenizer_json:
            raise ValueError("Checkpoint tokenizer differs from this run")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scaler.is_enabled() and resume["scaler"]:
            scaler.load_state_dict(resume["scaler"])
        start_epoch, next_batch = resume["epoch"], resume["next_batch"]
        global_step = resume["global_step"]
        best_validation_loss = resume["best_validation_loss"]
        torch.set_rng_state(resume["rng_state"])
        if device.type == "cuda" and resume["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        print(f"Resuming epoch {start_epoch + 1}, batch {next_batch}, step {global_step}")

    def persist(epoch, batch, *, best=False):
        if checkpoint_dir is None:
            return
        state = {
            "config": config_dict(config), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "optimizer_type": type(optimizer).__name__,
            "scaler": scaler.state_dict(), "epoch": epoch, "next_batch": batch,
            "global_step": global_step, "best_validation_loss": best_validation_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "tokenizer_json": tokenizer_json, "data_info": data_info,
        }
        save_checkpoint(Path(checkpoint_dir) / "latest.pt", state)
        if best:
            save_checkpoint(Path(checkpoint_dir) / "best.pt", state)

    dataLoader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        # Isolate DataLoader's iterator seed from model dropout when resuming.
        generator=torch.Generator().manual_seed(0),
    )

    model.train()  # Set the model to training mode
    print("Starting training...")

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss, total_tokens = 0.0, 0
        for step, (x, y) in enumerate(dataLoader):
            if epoch == start_epoch and step < next_batch:
                continue
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Clear previous gradients

            with autocast(device_type=device.type, enabled=device.type == "cuda"):  # Mixed precision context
                logits, loss = model(x, y)  # Forward pass and loss computation

            scaler.scale(loss).backward()  # Backpropagation with scaled loss

            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)  # Update parameters
            scaler.update()  # Update the scale for next iteration
            global_step += 1
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss Value: {loss.item():.4f}")
            if global_step % save_every == 0:
                persist(epoch, step + 1)

        if total_tokens:
            print(f"Epoch {epoch + 1} trained-token mean loss: {total_loss / total_tokens:.4f}")
        improved = False
        if validation_dataset is not None:
            metrics = evaluate_model(model, validation_dataset, config.batch_size)
            print(f"Validation loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")
            improved = metrics["loss"] < best_validation_loss
            best_validation_loss = min(best_validation_loss, metrics["loss"])
        persist(epoch + 1, 0, best=improved)
