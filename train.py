import torch
import torch.nn as nn
import os

from model_conf import ModelConfig

device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

torch.set_float32_matmul_precision("high")

class TransformerBlock(nn.Module):
    """ 
    A single transformer block consisting of a multi-head attention layer and a feed-forward network.
    args: 
        config: ModelConfig object containing model hyperparameters.
        x: Tensor of shape (batch_size, block_size, n_embd) representing the input to the transformer block.
    
    outputs:
        x: Tensor of shape (batch_size, block_size, n_embd) after passing through the transformer block.

    """ 
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Multi-head attention layer using pyTorch's optimized math
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # feed-forward network (MLP) with GELU activation and dropout for regularization
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        # pyTorch dot product attention for speed
        # automatically triggers FlashAttention
        norm_x = self.ln_1(x)

        # Self-Attention pyTorch logic (layers)
        x = x + self.mlp(self.ln_2(x))
        return x

