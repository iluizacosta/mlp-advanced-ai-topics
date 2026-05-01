import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron for multiclass classification.

    Architecture:
    Input -> [Linear -> ReLU -> Dropout] x N -> Linear (output)
    """

    def __init__(self, input_dim, hidden_sizes, output_dim, dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        # Output layer (NO activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)