# models/qnet_rnn.py (建議將 RNN 版本放在一個新文件或清晰命名)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# NoisyLinear 類的定義保持不變 (您可以從舊的 qnet.py 中複製過來)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def _scale_noise(self, size): # Helper for a bit cleaner reset_noise
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in)) # Outer product
        self.bias_epsilon.copy_(eps_out) # Use scaled noise for bias too

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


class QNetRNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=3,
                 # --- 可調整的潛力參數 ---
                 feature_dim=128,       # 特徵提取層的輸出維度
                 lstm_hidden_dim=128,   # LSTM 隱藏層維度
                 lstm_layers=1,         # LSTM 層數 (通常1-2層)
                 head_hidden_dim=128    # Dueling Head 前的共享隱藏層 (可選)
                 # --------------------------
                ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.head_hidden_dim = head_hidden_dim

        # 1. 特徵提取層 (處理單幀觀測)
        # 相較於原版 QNet 的 64 維，這裡可以考慮稍微增加維度以提供更豐富的特徵給 LSTM
        self.features_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim // 2), # 例如 7 -> 64
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),      # 例如 64 -> 128
            nn.ReLU()
        )

        # 2. LSTM 層 (處理特徵序列)
        # input_size 是 features_extractor 的輸出維度
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True) # batch_first=True 表示輸入維度為 (batch, seq_len, feature_dim)

        # 3. Dueling DQN 的頭部
        # 可以在 LSTM 輸出後再接一個小的共享 MLP 層，然後再分到 V 和 A
        # 這是可選的，但有時能幫助整合 LSTM 的輸出
        if self.head_hidden_dim > 0:
            self.fc_shared_head = nn.Sequential(
                NoisyLinear(lstm_hidden_dim, head_hidden_dim),
                nn.ReLU()
            )
            adv_value_input_dim = head_hidden_dim
        else:
            self.fc_shared_head = None # 也可以選擇不要這個共享層
            adv_value_input_dim = lstm_hidden_dim

        self.fc_V = NoisyLinear(adv_value_input_dim, 1) # 價值流
        self.fc_A = NoisyLinear(adv_value_input_dim, output_dim) # 優勢流

    def reset_noise(self):
        """重置所有 NoisyLinear 層的噪聲。"""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x_sequence, hidden_state_tuple): # <--- 修改這裡
        """
        處理觀測序列並返回 Q 值和下一個隱藏狀態。
        Args:
            x_sequence (Tensor): 輸入的觀測序列，形狀 (batch_size, sequence_length, input_dim)
            hidden_state_tuple (tuple): LSTM 的上一個隱藏狀態 (h_prev, c_prev)
                                       h_prev, c_prev 的形狀均為 (num_lstm_layers, batch_size, lstm_hidden_dim)
        Returns:
            q_values (Tensor): 動作的 Q 值，形狀 (batch_size, output_dim)。
            next_hidden_state_tuple (tuple): LSTM 的下一個隱藏狀態 (h_n, c_n)
        """
        batch_size, seq_len, _ = x_sequence.shape
        # h_prev, c_prev = hidden_state_tuple # 如果需要在內部解包，但可以直接傳遞元組給 LSTM

        # (1) 特徵提取
        x_flat = x_sequence.reshape(batch_size * seq_len, self.input_dim)
        features_flat = self.features_extractor(x_flat)
        features_sequence = features_flat.reshape(batch_size, seq_len, self.feature_dim)

        # (2) LSTM 層處理序列
        # lstm_output shape: (batch_size, sequence_length, lstm_hidden_dim)
        # h_n, c_n shape: (num_lstm_layers, batch_size, lstm_hidden_dim)
        # 直接將 hidden_state_tuple 傳給 LSTM
        lstm_output, (h_n, c_n) = self.lstm(features_sequence, hidden_state_tuple)

        # (3) 通常我們用 LSTM 序列的最後一個時間步的輸出來做決策
        last_lstm_output = lstm_output[:, -1, :] # Shape: (batch_size, lstm_hidden_dim)

        # (4) Dueling Heads
        x = last_lstm_output
        if self.fc_shared_head is not None:
            x = self.fc_shared_head(x)

        V = self.fc_V(x)
        A = self.fc_A(x)
        q_values = V + (A - A.mean(dim=1, keepdim=True))

        return q_values, (h_n, c_n)

    def init_hidden(self, batch_size, device):
        """為 LSTM 初始化隱藏狀態和細胞狀態 (通常為全零)。"""
        # Shape: (num_layers * num_directions, batch, hidden_size)
        # LSTM 預設是單向的，所以 num_directions = 1
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(device)
        return (h0, c0)