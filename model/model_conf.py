

class ModelConfig:
    vocab_size = 8000
    block_size = 512    # Maximum Context Length
    n_embd = 384
    n_head = 6          # Number of attention heads (384 / 6 = 64, so each head has 64 dimensions)
    n_layer = 6
    dropout = 0.1       # Prevent overfitting
    batch_size = 64     # Number of chunks per batch
