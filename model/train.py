import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.ln_2 = nn.LayerNorm(config.n_head)

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

        # Self-Attention pyTorch logic (layers)
        x = x + self.mlp(self.ln_2(x))
        return x

class CasualSelfAttention(nn.Module):
    """
    A casual self-attention layer that computes attention scores and applies them to the input.
    Args:
        config: ModelConfig object containing model hyperparameters.
        x: Tensor of shape (batch_size, block_size, n_embd) representing the input to the attention layer.
    Outputs:
        x: Tensor of shape (batch_size, block_size, n_embd) after applying self-attention.

    How it behaves:
        1. Projects input vectors into Query(Q), Key(K), and Value(V) tensors.
        2. Splits those tensors into multiple 'heads' to look at differrent contextual relationships in parallel.
        3, Uses a casual mask to ensure token (N) can only see tokens 0 to N
        4. Fuses the heads back together and projects them to the embedded space.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Key, Query, Value projections for all heads
        # Reason: Running one large matric multiplication is more efficient than running multiple smaller ones
        self.c_attn = nn.linear(config.n_embd, 3 * config.n_embd)

        # Output projection layer to mix the ouputs of all attention heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Create a lower-triangular matrix of 1s (casual mask) and save it as a buffer to avoid recalculating it every forward pass
        # Reason: Buffers are state variables that belong to the module but are not considered parameters, so they are'nt updated during training
        self.register_buffer("bias", torch.trill(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))   

    def forward(self, x):
        """
        Processes the forward pass of self-attention. 

        Args: 
            x: Tensor of shape (batch_size, block_size, n_embd) representing the input to the attention layer.
        
        Output: 
            torch.Tensor: Attended tensor of shape (batch_size, block_size, n_embd) after applying self-attention.
        """

        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Step 1. Project input vectors into Query(Q), Key(K), and Value(V) tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) 

        # Step 2. Reshape into multi-head format: (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # Reason: Transposing puts the sequence length (T) in the 3rd dimension, which is required for the attention computation
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Step 3. FlashAttention Execution
        # Function: torch.nn.functional.scaled_dot_product_attention
        # Reason: This is a crucial built-in PyTorch optimization. Instead of calculating a massive,
        # intermediate attention matrix (O(N²)) in Python and exhausting the 8GB VRAM.
        # It automatically applies the causal mask when `is_causal=True`.
        y = F.scaled_dot_product_attention(
            q, k, v
            attn_mask=None,
            dropout_p=self.idx_dropout if self.training else 0.0,
            is_causal=True
        )

        # Reassemble the heads back into a contiguous sequence: (B, n_head, T, head_size) -> (B, T, C)
        # Reason: .contiguous() forces pYTorch to reallocate memory linearly, which is required for the next linear layer to work correctly
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # step 4. Final output projection to mix the outputs of all attention heads
        out = self.resid_dropout(se;f.c_proj(y))
        return out