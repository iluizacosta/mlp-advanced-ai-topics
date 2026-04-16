import numpy as np
import torch
import random

# Define a reproducibility seed

def set_seed(seed=42):
    """
    Define uma seed global para garantir reprodutibilidade.
    """

    # Seed para gerador aleatório do Python
    random.seed(seed)

    # Seed para operações com o NumPy
    np.random.seed(seed)

    # Seed para o PyTorch
    torch.manual_seed(seed)
    
    # Se houver GPU disponível, define seed também para CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)