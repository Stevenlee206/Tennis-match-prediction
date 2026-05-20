import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Residual Block tailored for Tabular Data.
    Uses BatchNorm, SiLU (Swish), and Dropout.
    """
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class TennisNet(nn.Module):
    """
    An improved Tabular ResNet model that preserves compatibility with
    the existing training pipeline but achieves significantly better classification performance.
    """
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.2):
        super(TennisNet, self).__init__()
        
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.last_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.last_layer(x)