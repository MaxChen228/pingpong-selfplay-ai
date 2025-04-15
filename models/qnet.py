import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, input_dim=7, output_dim=3):  # <-- 預設=7
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

