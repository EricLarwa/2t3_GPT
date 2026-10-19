import torch
import torch.nn as nn

from models.model_config import ModelConfig

class MLP(nn.module):
    """
    Multi-Layer perceptron (feed-Forward Network) that acts the model's "knowledge" layer. It takes the output of the attention layer and transforms it into a higher-dimensional space.
    Args:
        config: ModelConfig object containing model hyperparameters.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Function used: nn.GELU()
        # Reason: Gausssian Error Linear Ubits are smooth activation functions
        # better than ReLu, avoiding the 'dead nueron' problem 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gleu = nn.GELU()
        self.c_proj = nn.linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Porjects vectors 4 times wider, applies non-linear activation, then projects back to original size
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    """
    A single transformer block that consists of a multi-head attention layer followed by a feed-forward network (MLP).
    Args:
        config: ModelConfig object containing model hyperparameters.
        x: Tensor of shape (batch_size, block_size, n_embd) representing the input to the block.
    Outputs:
        x: Tensor of shape (batch_size, block_size, n_embd) after applying the block's operations.
    How it behaves:
        Applies a 'Residual' (Skip-connection) highway design. The input vectors pass around the 
        math operations and add back into the output. This allows gradients to travel thousands of layers
        without shrinking to zero (vanishing gradients).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # 1. Normalize -> Attend -> Add(Residual)
        x = x + self.attn(self.ln_1(x))
        # 2. Normalize -> MLP -> Add(Residual)
        x = x + self.mlp(self.ln_2(x))

        return x