
from torch.utils.data import Dataset, DataLoader

class TokenizerDataset(Dataset):
    """
    A high-performance PyTorch dataset that slices token sequences into context chunks.
    
    Args:
        token_ids (list): A massive, flat list of integers representing your entire text corpus.
        block_size (int): The context window length (e.g., 512 tokens).
        
    Reason:
        By storing the entire dataset as a single contiguous PyTorch LongTensor in the system RAM,
        we completely bypass disk I/O bottlenecks during training.
    """
    def __init__(self, data, token_ids, block_size):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Slices out a trraining sample input chunk (X) ands it's shifted
        target chunk (Y) for next-token prediction.

        Output: 
            x: Tensor of shape (block_size,) representing the input sequence.
            y: Tensor of shape (block_size,) representing the target sequence (next token prediction).
        """
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input sequence (context)
        y = chunk[1:]   # Target sequence (next token prediction)
        return x, y