
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model_conf import ModelConfig
from .build import Block

class LocalGPT(nn.Module):
    """
    The full autoregressive language model compiled from scratch.

    Args:
        config: ModelConfig object containing model hyperparameters.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.transformer = nn.ModuleDict(dict(
            # Toeken embedding: Maps tokens to vectors of size n_embd
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # Position Embedding: Learns a unique vector for index positions.
            # Reason: Transformers process all words at once. Without this, the model would not know the order of the words in a sentence.
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # Dropout layer to prevent overfitting
            drop = nn.Dropout(config.dropout),

            # Constructing a dynamic stack of layers based on our configuration
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final Normalization layer to ensure the output is well-behaved
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # Language Model head: Projects the final hidden states to the vocabulary size for token prediction
        # yields a tensor of shape (batch_size, block_size, vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Optimization: Weight Tying
        # Reason: Pointing the weights of the lm_head to the token embedding layer reduces the number of parameters and improves generalization.
        # This will save the local VRAM (roughlty ~#M variables in an 8,000 token vocabulary with 768 embedding size
        self.lm_head.weight = self.transformer.wte.weight

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, idx, targets=None):
        """
        Forward pass of the LocalGPT model.

        Args:
            idx: Tensor of shape (batch_size, block_size) containing token indices.
            targets: Optional tensor of shape (batch_size, block_size) containing target token indices for loss computation.

        Returns:
            If targets is provided, returns a tuple of (logits, loss).
            If targets is not provided, returns (logits, None).
        """
        # Get the batch size and sequence length from the input tensor
        device = idx.device
        B, T = idx.size()

        # Ensure the input sequence length does not exceed the model's block size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # absolute position range tensor
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token and position embeddings
        token_embeddings = self.transformer.wte(idx)  # Shape: (B, T, n_embd)
        position_embeddings = self.transformer.wpe(pos)  # Shape: (T, n_embd)

        # Combine token and position embeddings
        x = token_embeddings + position_embeddings.unsqueeze(0)  # Shape: (B, T, n_embd)

        # Apply dropout to the combined embeddings
        x = self.transformer.drop(x)

        # Pass through each transformer block
        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If training, calculate how wrong the predictions are.
            logits = self.lm_head(x)

            # Function used: F.cross_entropy()
            # Reason: We flatten the matrix from 3D to 2D because PyTorch loss targets expect
            # a simple sequence array to calculate math cross-references.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # Inference optimization: Only calculate logits for the final absolute token to save speed during generation.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
