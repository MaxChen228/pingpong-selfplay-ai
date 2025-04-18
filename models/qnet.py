import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, input_dim=7, output_dim=3):
        super(QNet, self).__init__()
        # 原先的 feature 提取层
        self.features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Dueling Head
        self.fc_V = nn.Linear(64, 1)
        self.fc_A = nn.Linear(64, output_dim)

    def forward(self, x):
        h = self.features(x)
        V = self.fc_V(h)  # [B,1]
        A = self.fc_A(h)  # [B,3]
        # Q = V + (A - A.mean())
        return V + (A - A.mean(dim=1, keepdim=True))
