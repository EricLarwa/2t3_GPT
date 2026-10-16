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
        