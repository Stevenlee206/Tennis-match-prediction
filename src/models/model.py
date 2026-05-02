import torch
import torch.nn as nn
class TennisNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(TennisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.net(x) # Trả về logits thô