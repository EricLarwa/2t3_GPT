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
                    "x_env_dir/Scripts/python.exe -m pip install bitsandbytes, then resume again"
                ) from error
            bnb = None
    else:
        bnb = None
    # bitsandbytes publicly re-exports AdamW8bit without an explicit typed re-export.
    optimizer_class = bnb.optim.AdamW8bit if bnb is not None else torch.optim.AdamW  # pyright: ignore[reportPrivateImportUsage]
    return optimizer_class(model.parameters(), lr=6e-4, weight_decay=0.1)

def decide_to_stop(epochs_without_improvement, patience=8):
    """
    Stop after consecutive validation checks without meaningful improvement.

    Args:
        epochs_without_improvement: Consecutive validation checks without improvement.
        patience: Number of checks to allow; zero disables early stopping.

    Returns:
        True if training should be stopped, False otherwise.
    """
    # Stop if the validation loss hasn't improved for a certain number of epochs
    # Count unsuccessful checks, rather than the absolute epoch number.
    return patience > 0 and epochs_without_improvement >= patience



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
                      save_every=100, early_stopping_patience=None,
                      early_stopping_min_delta=None, learning_rate=None):
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
        early_stopping_patience: Checks without improvement before stopping; default 8.
        early_stopping_min_delta: Minimum loss decrease to reset patience; default 0.01.
        learning_rate: Explicit override, applied after restoring optimizer state.
    """

    if epochs < 1 or save_every < 1:
        raise ValueError("epochs and save_every must be positive")
    if learning_rate is not None and (not 0 < learning_rate < float("inf")):
        raise ValueError("learning_rate must be finite and positive")
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

    signature = hashlib.sha256(memoryview(dataset.data.numpy())).hexdigest()
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

    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    reduce_learning_rate = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )
    if resume is not None:
        if "scheduler" in resume:
            reduce_learning_rate.load_state_dict(resume["scheduler"])
        elif best_validation_loss < float("inf"):
            # Older checkpoints have a validation baseline but no scheduler history.
            reduce_learning_rate.step(best_validation_loss)

    early_state = resume.get("early_stopping", {}) if resume is not None else {}
    patience = early_stopping_patience if early_stopping_patience is not None else early_state.get("patience", 8)
    min_delta = early_stopping_min_delta if early_stopping_min_delta is not None else early_state.get("min_delta", 0.01)
    if patience < 0 or not 0 <= min_delta < float("inf"):
        raise ValueError("Early stopping patience and min_delta must be non-negative and finite")
    early_best_loss = early_state.get("best_loss", best_validation_loss)
    bad_epochs = early_state.get("bad_epochs", 0)

    def persist(epoch, batch, *, best=False):
        if checkpoint_dir is None:
            return
        state = {
            "config": config_dict(config), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "optimizer_type": type(optimizer).__name__,
            "scaler": scaler.state_dict(), "epoch": epoch, "next_batch": batch,
            "scheduler": reduce_learning_rate.state_dict(),
            "early_stopping": {"best_loss": early_best_loss, "bad_epochs": bad_epochs,
                               "patience": patience, "min_delta": min_delta},
            "global_step": global_step, "best_validation_loss": best_validation_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "tokenizer_json": tokenizer_json, "data_info": data_info,
        }
        save_checkpoint(Path(checkpoint_dir) / "latest.pt", state)
        if best:
            save_checkpoint(Path(checkpoint_dir) / "best.pt", state)

    def restore_best():
        if checkpoint_dir is not None and validation_dataset is not None:
            best_path = Path(checkpoint_dir) / "best.pt"
            if best_path.is_file():
                from .checkpoint import load_checkpoint

                best = load_checkpoint(best_path)
                if best["data_info"] != data_info or best["config"] != config_dict(config):
                    raise ValueError("Best checkpoint belongs to different training data or configuration")
                model.load_state_dict(best["model"])
                print("Loaded best validation weights for final evaluation; latest.pt retains resume state")

    if validation_dataset is not None and decide_to_stop(bad_epochs, patience):
        print("Checkpoint already reached early stopping; use --patience 0 to continue explicitly")
        restore_best()
        return

    dataLoader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
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
        stop = False
        if validation_dataset is not None:
            metrics = evaluate_model(model, validation_dataset, config.batch_size)
            print(f"Validation loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")
            improved = metrics["loss"] < best_validation_loss
            best_validation_loss = min(best_validation_loss, metrics["loss"])
            reduce_learning_rate.step(metrics["loss"])  # Adjust learning rate based on validation loss
            if metrics["loss"] < early_best_loss - min_delta:
                early_best_loss, bad_epochs = metrics["loss"], 0
            else:
                bad_epochs += 1
            stop = decide_to_stop(bad_epochs, patience)
            print(f"Validation checks without meaningful improvement: {bad_epochs}/{patience or 'disabled'}")
            print(f"Learning rate for next epoch: {optimizer.param_groups[0]['lr']:.6g}")
        persist(epoch + 1, 0, best=improved)
        if stop:
            print("Early stopping triggered. Ending training.")
            break
    restore_best()
