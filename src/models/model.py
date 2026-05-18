import torch
import torch.nn as nn

class ReshapeInput(nn.Module): #nếu dùng Conv
    def __init__(self):
        super(ReshapeInput, self).__init__()

    def forward(self, x):
        return x.unsqueeze(1)

class TennisNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(TennisNet, self).__init__()
        self.net = nn.Sequential(
            # --- BIẾN ĐỔI ĐẦU VÀO ---
            ReshapeInput(),

            # --- KHỐI TÍCH CHẬP 1 (Channels: 1 -> 2 -> 4) ---
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.35),

            nn.MaxPool1d(kernel_size=2),

            # --- KHỐI TÍCH CHẬP 2 (Channels: 4 -> 8 -> 16) ---
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Dropout(0.33),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.31),

            nn.MaxPool1d(kernel_size=2),

            # --- KHỐI TÍCH CHẬP 3 & GLOBAL POOLING (Channels: 16 -> 32 -> 64) ---
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # --- KHỐI TUYẾN TÍNH PHÂN LOẠI (2 LỚP LINEAR CUỐI) ---
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, 2)
        )
    def forward(self, x):
        return self.net(x) # Trả về logits thô