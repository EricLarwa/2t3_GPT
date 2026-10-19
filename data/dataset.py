
import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    """
    A high-performance PyTorch dataset that slices token sequences into context chunks.

    Args:
        token_ids (list or Tensor): A flat list or memory-mapped tensor of corpus token IDs.
        block_size (int): The context window length (e.g., 512 tokens).
        stride (int): Distance between chunk starts; use block_size for sequential contexts.

    Reason:
        By storing the entire dataset as a single contiguous PyTorch LongTensor in the system RAM,
        we completely bypass disk I/O bottlenecks during training.
    """
    def __init__(self, token_ids, block_size, stride=1):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if len(token_ids) <= block_size:
            raise ValueError(f"Corpus must contain at least {block_size + 1} tokens")
        if stride < 1:
            raise ValueError("stride must be positive")
        self.data = torch.as_tensor(token_ids, dtype=torch.long).contiguous()
        self.block_size = block_size
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.block_size - 1) // self.stride + 1

    def __getitem__(self, idx):
        """
        Slices out a trraining sample input chunk (X) ands it's shifted
        target chunk (Y) for next-token prediction.

        Output:
            x: Tensor of shape (block_size,) representing the input sequence.
            y: Tensor of shape (block_size,) representing the target sequence (next token prediction).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Dataset index out of range")
        start = idx * self.stride
        chunk = self.data[start:start + self.block_size + 1]
        x = chunk[:-1]  # Input sequence (context)
        y = chunk[1:]   # Target sequence (next token prediction)
        return x, y
