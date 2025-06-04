# train_rnn_iterative.py (新文件)

import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import copy
import time

# 假設您的 QNetRNN 定義在 models/qnet_rnn.py
from models.qnet_rnn import QNetRNN
from models.qnet import QNet # 可能仍需要 QNet 用於加載舊的對手池模型
from envs.my_pong_env_2p import PongEnv2P
import torch.nn.functional as F

# ------------------- 限制線程數 (可選，與原文件一致) -------------------
os.environ['OMP_NUM_THREADS']    = '8'
os.environ['MKL_NUM_THREADS']    = '8'
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# ------------------- 讀取配置 (與原文件一致，但可能需要為 RNN 添加新配置) ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
def get_cfg(key, default=None):
    return cfg['training'].get(key, default)

# --- RNN 相關的新配置 (需要在 config.yaml 中添加) ---
# 例如:
# training:
#   model_type: "QNetRNN" # 或 "QNet"
#   feature_dim: 128
#   lstm_hidden_dim: 128
#   lstm_layers: 1
#   head_hidden_dim: 128
#   trace_length: 8         # 訓練時使用的序列長度
#   burn_in_length: 4       # (可選) BPTT burn-in 長度
#   rnn_save_model_prefix: "model_rnn_" # RNN 模型保存前綴
#   rnn_init_model_path: "checkpoints_rnn/initial_rnn_model.pth" # RNN 初始模型 (可選)

model_type = get_cfg('model_type', 'QNetRNN') # 默認使用 RNN
feature_dim = get_cfg('feature_dim', 128)
lstm_hidden_dim = get_cfg('lstm_hidden_dim', 128)
lstm_layers = get_cfg('lstm_layers', 1)
head_hidden_dim = get_cfg('head_hidden_dim', 128)
trace_length = get_cfg('trace_length', 8) # 重要：訓練時的序列長度
# burn_in_length = get_cfg('burn_in_length', 0) # 如果使用 burn-in

# --- 其他訓練超參 (與原文件一致或微調) ---
max_generations         = get_cfg('max_generations')
episodes_per_generation = get_cfg('episodes_per_generation')
eval_episodes           = get_cfg('eval_episodes')
curr_win_threshold      = get_cfg('curr_win_threshold')
pool_win_threshold      = get_cfg('pool_win_threshold')
lr                      = get_cfg('lr')
gamma                   = get_cfg('gamma')
batch_size              = get_cfg('batch_size') # 對於 RNN，這通常是序列的批次大小
memory_size             = get_cfg('memory_size') # Replay Buffer 大小
epsilon_decay           = get_cfg('epsilon_decay')
min_epsilon             = get_cfg('min_epsilon')

# 針對 RNN 的模型路徑和ID
rnn_model_id_prefix = get_cfg('model_id_prefix', 'rnn_agent_2_') # 用於區分模型
init_model_path     = get_cfg('init_model_path_rnn', None) # RNN 的初始模型路徑，可以是 None
                                                            # 如果為 None，則從隨機初始化開始
                                                            # 注意：不能直接用舊 QNet 的 .pth
opponent_pool_ratio = get_cfg('opponent_pool_ratio')
target_update_interval= get_cfg('target_update_interval')
win_rate_interval       = get_cfg('win_rate_interval')
max_retries             = get_cfg('max_retries_for_generation')

ckpt_dir = get_cfg('ckpt_dir_rnn', 'checkpoints_rnn') # RNN 模型的保存目錄
os.makedirs(ckpt_dir, exist_ok=True)
plot_dir = get_cfg('plot_dir_rnn', 'plot_rnn')
os.makedirs(plot_dir, exist_ok=True)


# ------------------- 新的序列經驗回放池 (PrioritizedReplay 需要大改或替換) ---
# 這是最複雜的部分。一個簡化的思路是存儲 episode traces，然後從中採樣。
# 或者，修改 PrioritizedReplay 以適應序列。
# 為了快速開始，我們先定義一個簡單的 SequenceReplayBuffer (非 PER)
# 之後可以再考慮如何將其與 PER 結合或使用更高級的序列 Replay Buffer。

class SequenceReplayBuffer:
    def __init__(self, capacity, trace_length):
        self.capacity = capacity
        self.trace_length = trace_length
        self.buffer = deque(maxlen=capacity) # 存儲完整的 episode trajectories
        self.current_episode_trajectory = [] # 用於構建當前 episode

    def push_step(self, state, action, reward, next_state, done, hidden_state_actor, hidden_state_critic_target):
        # 存儲單步 transition，包含隱藏狀態以便於後續處理 (如果需要)
        # 實際上，對於 DRQN，隱藏狀態通常在訓練時重新計算 (burn-in)
        # 因此，這裡主要存儲 (s, a, r, s', done)
        # hidden_state 通常不在 replay buffer 中存儲以節省空間並允許 batching
        self.current_episode_trajectory.append((state, action, reward, next_state, done))
        if done:
            if len(self.current_episode_trajectory) >= self.trace_length: # 只有當 episode 長度足夠時才存儲
                self.buffer.append(list(self.current_episode_trajectory)) # 存儲副本
            self.current_episode_trajectory = []

    def sample(self, batch_size):
        # 確保 buffer 中至少有一個完整的 episode，並且 episode 數量足以進行採樣 (或允許重複採樣)
        if len(self.buffer) == 0:
            # print("[DEBUG ReplayBuffer] Buffer is empty, cannot sample.")
            return None # 返回 None 表示無法採樣

        num_available_episodes = len(self.buffer)
        
        # 如果可用 episodes 數量少於 batch_size，則允許重複採樣 episodes
        # 或者，如果希望每個 batch 中的 trace 來自不同的 episode (如果可能)，則需要更複雜的邏輯
        # 這裡我們允許重複採樣 episode 來簡化
        sampled_ep_indices = np.random.choice(num_available_episodes, batch_size, replace=True)

        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []
        actual_samples_count = 0

        for ep_idx in sampled_ep_indices:
            episode = self.buffer[ep_idx]
            
            # 確保 episode 長度足以提取一個 trace
            if len(episode) < self.trace_length:
                # print(f"[DEBUG ReplayBuffer] Episode {ep_idx} is too short (len {len(episode)}) for trace_length {self.trace_length}. Skipping.")
                continue # 跳過這個太短的 episode

            # 從 episode 中隨機選擇一個起始點以形成一個 trace_length 的序列
            start_idx_in_episode = np.random.randint(0, len(episode) - self.trace_length + 1)
            trace = episode[start_idx_in_episode : start_idx_in_episode + self.trace_length]
            
            # 解包 trace 中的數據
            # obs, action, reward, next_obs, done
            obs_s, act_s, rew_s, next_obs_s, done_s_flags = zip(*trace)
            
            obs_batch.append(np.array(obs_s, dtype=np.float32)) # (trace_length, obs_dim)
            act_batch.append(np.array(act_s, dtype=np.int64))   # (trace_length,)
            rew_batch.append(np.array(rew_s, dtype=np.float32)) # (trace_length,)
            next_obs_batch.append(np.array(next_obs_s, dtype=np.float32)) # (trace_length, obs_dim)
            done_batch.append(np.array(done_s_flags, dtype=np.bool_))    # (trace_length,)
            actual_samples_count +=1

        if actual_samples_count == 0: # 如果因為所有 episode 都太短而沒有採集到任何樣本
            # print("[DEBUG ReplayBuffer] No valid traces could be sampled.")
            return None

        # 如果實際採集到的樣本數少於要求的 batch_size (雖然上面 replace=True 應該能保證數量，但以防萬一)
        # 這裡我們還是以 actual_samples_count 為準來創建張量
        # 如果您希望嚴格的 batch_size，那麼在 episode 過短時的處理需要更魯棒 (例如循環直到取夠)

        return (
            torch.tensor(np.array(obs_batch), dtype=torch.float32, device=device),    # (batch_size, trace_length, obs_dim)
            torch.tensor(np.array(act_batch), dtype=torch.long, device=device),       # (batch_size, trace_length)
            torch.tensor(np.array(rew_batch), dtype=torch.float32, device=device),   # (batch_size, trace_length)
            torch.tensor(np.array(next_obs_batch), dtype=torch.float32, device=device),# (batch_size, trace_length, obs_dim)
            torch.tensor(np.array(done_batch), dtype=torch.bool, device=device),      # (batch_size, trace_length)
            None, # sampled_indices (for PER, 暫時為 None)
            None  # importance_weights (for PER, 暫時為 None)
        )

    def __len__(self):
        return len(self.buffer) # 返回存儲的 episode 數量


# ------------------- 環境 & 設備 -------------------
env    = PongEnv2P(**cfg['env'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ------------------- 模型創建函數 (已在上方修改為 QNetRNN) -------------------
def create_qnet_rnn_model():
    # 獲取觀測維度
    obs_shape = env.observation_space.shape
    if obs_shape is None or not hasattr(env.observation_space, 'shape'): # 額外保護
        input_dim_val = 7 # 默認值
        print(f"[Warning] Could not determine input_dim from env.observation_space.shape, using default {input_dim_val}.")
    elif len(obs_shape) == 1: # 例如 Box((7,), ...) -> shape is (7,)
        input_dim_val = obs_shape[0]
    elif len(obs_shape) > 1: # 例如 Box(..., shape=(H, W, C)) -> shape is (H,W,C) -> 可能需要展平或其他處理
        # 對於 PongEnv2P，我們期望的是扁平的 (7,)
        # 如果是圖像等高維觀測，這裡需要不同的處理邏輯
        # 但根據 PongEnv2P 的定義，應該是 (7,)
        input_dim_val = obs_shape[0] # 假設我們仍然取第一個維度，或者需要一個展平後的總維度 np.prod(obs_shape)
        print(f"[Warning] env.observation_space.shape is {obs_shape}, using first dimension {input_dim_val} as input_dim. Verify if this is correct for multi-dimensional input.")
    else: # 空的 shape 元組
        input_dim_val = 7
        print(f"[Warning] env.observation_space.shape is empty, using default {input_dim_val}.")


    # 獲取動作維度 (這部分之前的邏輯應該是正確的)
    if hasattr(env.action_space, 'nvec'): # MultiDiscrete
        output_dim_val = env.action_space.nvec[0] # 假設我們只關心第一個離散動作空間的大小
    elif hasattr(env.action_space, 'n'): # Discrete
        output_dim_val = env.action_space.n
    else:
        output_dim_val = 3 # 默認值
        print(f"[Warning] Could not determine output_dim from env.action_space, using default {output_dim_val}.")

    print(f"[Debug] Creating QNetRNN with input_dim={input_dim_val}, output_dim={output_dim_val}")
    return QNetRNN(input_dim=input_dim_val,
                   output_dim=output_dim_val,
                   feature_dim=feature_dim,
                   lstm_hidden_dim=lstm_hidden_dim,
                   lstm_layers=lstm_layers,
                   head_hidden_dim=head_hidden_dim).to(device)

# ------------------- 加載或初始化模型 (針對 RNN) -------------------
if init_model_path and os.path.exists(init_model_path):
    print(f"[INFO] Loading initial RNN model from {init_model_path}")
    checkpoint = torch.load(init_model_path, map_location=device)
    # 假設 checkpoint 格式與之前類似，但存儲的是 RNN 模型的參數
    modelA_state = checkpoint.get('modelA_state', checkpoint.get('modelB_state', checkpoint.get('model')))
    modelB_state = checkpoint.get('modelB_state', checkpoint.get('modelA_state', checkpoint.get('model'))) # 或者 modelA 和 B 分別初始化

    modelA = create_qnet_rnn_model()
    if modelA_state: modelA.load_state_dict(modelA_state)
    modelB = create_qnet_rnn_model()
    if modelB_state: modelB.load_state_dict(modelB_state)
    
    # old_state 用於 reset_B 時恢復 modelB 的初始狀態 (可以是 modelA 的狀態或一個特定的初始 RNN 狀態)
    old_state_for_reset = copy.deepcopy(modelA.state_dict()) # 或者一個預存的初始 RNN 狀態

    epsilon = checkpoint.get('epsilon', 1.0)
    global_episode_count = checkpoint.get('episode', 0)
    # done_generations = checkpoint.get('generation', 0) # 如果需要從特定 generation 恢復
else:
    print("[INFO] Initializing new RNN models randomly.")
    modelA = create_qnet_rnn_model()
    modelB = create_qnet_rnn_model()
    old_state_for_reset = copy.deepcopy(modelA.state_dict()) # 保存 modelA 的初始狀態用於 reset_B
    epsilon = 1.0
    global_episode_count = 0
    # done_generations = 0

print(f"[INFO] model_type: {model_type}")
print(f"[INFO] Model A ({'RNN' if isinstance(modelA, QNetRNN) else 'QNet'}) initialized.")
print(f"[INFO] Model B ({'RNN' if isinstance(modelB, QNetRNN) else 'QNet'}) initialized, training target.")

# 凍結 modelA (對手) 的參數
for p in modelA.parameters(): p.requires_grad = False
modelA.eval() # 確保 modelA 處於評估模式 (影響 NoisyLinear, Dropout 等)

# ------------------- 目標網路 & 優化器 (針對 RNN modelB) -------------------
targetB = create_qnet_rnn_model()
targetB.load_state_dict(modelB.state_dict())
targetB.eval() # 目標網路也應處於評_估模式

optimizerB = optim.Adam(modelB.parameters(), lr=lr) # modelB 的所有參數都參與訓練

# ------------------- 初始化 Replay Buffer、記錄與計時 -------------------
# memory = PrioritizedReplay(memory_size, alpha=0.6) # 舊的
memory = SequenceReplayBuffer(memory_size, trace_length) # 新的序列 Replay Buffer
# PER 的 beta 參數 (如果使用 PER)
# beta_start     = 0.4
# beta_frames    = 100000
# frame_idx_for_beta = 0 # 用於 beta 退火

reward_history = [] # 記錄每個 episode modelB 的獎勵
winA_deque     = deque(maxlen=win_rate_interval) # 對戰 modelA 的勝率隊列
winP_deque     = deque(maxlen=win_rate_interval) # 對戰 Pool 的勝率隊列
train_steps_count = 0

start_time     = time.time()
interval_start_time = start_time

# ------------------- 動作選擇 (已在上一回復中提供思路，需整合) ------------
def select_action_for_model(model_instance, obs_single_frame, h_prev, c_prev, current_epsilon, is_eval_mode=False):
    """通用動作選擇函數，適用於 RNN 模型"""
    hidden_input_tuple = (h_prev, c_prev) # <--- 打包隱藏狀態

    if not is_eval_mode and random.random() < current_epsilon: # 探索
        action = random.randint(0, model_instance.fc_A.out_features - 1)
        with torch.no_grad():
            obs_seq = torch.tensor(obs_single_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            # 調用時傳遞元組
            _, (h_next, c_next) = model_instance(obs_seq, hidden_input_tuple) # <--- 修改這裡
    else: # 利用
        with torch.no_grad():
            if hasattr(model_instance, 'reset_noise') and not is_eval_mode:
                model_instance.reset_noise()
            obs_seq = torch.tensor(obs_single_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            # 調用時傳遞元組
            q_values, (h_next, c_next) = model_instance(obs_seq, hidden_input_tuple) # <--- 修改這裡
            action = q_values.argmax(1).item()
    return action, (h_next, c_next)


# ------------------- 訓練步驟 (train_step) - 需要為 RNN 大改 ---------------
# train_rnn_iterative.py

# (確保 torch.nn.functional 已導入: import torch.nn.functional as F)

# (burn_in_length 的定義，如果需要的話，可以從 config 讀取)
# burn_in_length = get_cfg('burn_in_length', 0) # 默認為0，即不使用 burn-in

def train_step_rnn():
    global train_steps_count, epsilon # epsilon 用於打印日誌

    # 確保 Replay Buffer 中有足夠的 episodes 來採樣 batch_size 個 traces
    # min_episodes_for_training Factor 來自您的 config 或一個合理的預設值
    min_episodes_to_start = batch_size * get_cfg('min_episodes_for_training_start_factor', 1)
    if len(memory) < min_episodes_to_start :
        # print(f"[DEBUG train_step] Not enough episodes in memory ({len(memory)}/{min_episodes_to_start}) to sample a batch.")
        return

    # 從 Replay Buffer 採樣序列數據
    # sampled_data = (obs_sequences, action_sequences, reward_sequences, next_obs_sequences, done_sequences, sampled_indices, importance_weights)
    sampled_data = memory.sample(batch_size)
    if sampled_data is None or sampled_data[0] is None: # 確保 sample 成功返回數據
        # print("[DEBUG train_step] memory.sample() returned None.")
        return
        
    obs_sequences, action_sequences, reward_sequences, \
    next_obs_sequences, done_sequences, \
    sampled_indices, importance_weights = sampled_data # sampled_indices 和 importance_weights 來自 SimpleSequenceReplayBuffer，目前是 None

    current_actual_batch_size = obs_sequences.size(0) # 實際採樣到的 batch 大小
    if current_actual_batch_size == 0: # 以防萬一
        return

    # --- 準備 LSTM 的初始隱藏狀態 (全零) ---
    # 對於每個序列，其 LSTM 計算都從一個初始的零隱藏狀態開始。
    # 如果使用了 burn-in，這個 h0, c0 會被 burn-in 階段的輸出所取代。
    h0_modelB, c0_modelB = modelB.init_hidden(current_actual_batch_size, device)
    h0_targetB, c0_targetB = targetB.init_hidden(current_actual_batch_size, device)
    
    # --- (可選) Burn-in 階段 (這裡我們不實現，但保留思路) ---
    # if burn_in_length > 0 and trace_length > burn_in_length:
    #     with torch.no_grad():
    #         # Burn-in for modelB path
    #         _, (h_burn_modelB, c_burn_modelB) = modelB(obs_sequences[:, :burn_in_length, :], (h0_modelB, c0_modelB))
    #         # Burn-in for targetB path (using next_obs for consistency if targets are from next_obs)
    #         _, (h_burn_targetB, c_burn_targetB) = targetB(next_obs_sequences[:, :burn_in_length, :], (h0_targetB, c0_targetB)) # 或 modelB
            
    #     # 使用 burn-in 後的隱藏狀態作為後續計算的初始狀態
    #     h_start_modelB, c_start_modelB = h_burn_modelB, c_burn_modelB
    #     h_start_targetB, c_start_targetB = h_burn_targetB, c_burn_targetB
        
    #     # 截取 burn-in 之後的序列用於訓練
    #     train_obs_seq = obs_sequences[:, burn_in_length:, :]
    #     train_act_seq = action_sequences[:, burn_in_length:]
    #     train_rew_seq = reward_sequences[:, burn_in_length:]
    #     train_next_obs_seq = next_obs_sequences[:, burn_in_length:, :]
    #     train_done_seq = done_sequences[:, burn_in_length:]
    # else: # 不使用 burn-in，直接使用整個序列
    h_start_modelB, c_start_modelB = h0_modelB, c0_modelB
    h_start_targetB, c_start_targetB = h0_targetB, c0_targetB
    train_obs_seq = obs_sequences
    train_act_seq = action_sequences
    train_rew_seq = reward_sequences
    train_next_obs_seq = next_obs_sequences
    train_done_seq = done_sequences


    # --- 1. 計算當前 Q 值: Q(s_t, a_t) ---
    # modelB 的 forward 方法返回序列最後一個時間步的 Q 值和最後的隱藏狀態
    # q_values_all_actions_last_step: (batch_size, num_actions)
    q_values_all_actions_last_step, _ = modelB(train_obs_seq, (h_start_modelB, c_start_modelB))
    
    # action_sequences: (batch_size, trace_length)
    # 我們需要的是在序列的最後一個時間步 (T-1) 實際執行的動作。
    # train_act_seq 的最後一列是我們要找的動作。
    actions_at_T_minus_1 = train_act_seq[:, -1].unsqueeze(1) # Shape: (batch_size, 1)
    
    # 從 Q(s_T-1, :) 中提取 Q(s_T-1, a_T-1)
    q_values_for_action_at_T_minus_1 = q_values_all_actions_last_step.gather(1, actions_at_T_minus_1).squeeze(1) # Shape: (batch_size,)

    # --- 2. 計算目標 Q 值: r_{T-1} + gamma * max_a Q_target(s_T, a) ---
    # s_T 是 train_next_obs_seq 的最後一列所代表的狀態
    # r_{T-1} 是 train_rew_seq 的最後一列
    # done_{T-1} 是 train_done_seq 的最後一列 (表示 s_T 是否為終止狀態)
    with torch.no_grad():
        # Double DQN:
        # a'. 使用 modelB 網絡（當前策略網絡）為 next_obs_sequences（即 s_1, ..., s_T）的最後一個狀態 s_T 選擇最佳動作 a*
        # q_next_all_actions_from_modelB_last_step: (batch_size, num_actions)
        q_next_all_actions_from_modelB_last_step, _ = modelB(train_next_obs_seq, (h_start_targetB, c_start_targetB)) # 使用 targetB 的初始隱藏狀態流
        best_next_actions = q_next_all_actions_from_modelB_last_step.argmax(dim=1, keepdim=True) # Shape: (batch_size, 1)

        # b'. 使用 targetB 網絡評估這些最佳動作 a* 在狀態 s_T 的 Q 值
        # q_next_all_actions_from_targetB_last_step: (batch_size, num_actions)
        q_next_all_actions_from_targetB_last_step, _ = targetB(train_next_obs_seq, (h_start_targetB, c_start_targetB))
        
        # 從 Q_target(s_T, :) 中提取 Q_target(s_T, a*)
        q_next_target_values = q_next_all_actions_from_targetB_last_step.gather(1, best_next_actions).squeeze(1) # Shape: (batch_size,)

        # 提取序列最後一步的獎勵和完成標記
        # train_rew_seq 和 train_done_seq 的形狀是 (batch_size, effective_trace_length)
        rewards_at_T_minus_1 = train_rew_seq[:, -1] # Shape: (batch_size,)
        dones_at_T_minus_1 = train_done_seq[:, -1]   # Shape: (batch_size,) (表示 s_T 是否是終止狀態)
        
        # 計算 TD 目標
        td_targets = rewards_at_T_minus_1 + gamma * q_next_target_values * (~dones_at_T_minus_1) # ~dones_at_T_minus_1 是 (1 - done)

    # --- 3. 計算損失 ---
    # 使用 Smooth L1 Loss (Huber loss)
    loss = F.smooth_l1_loss(q_values_for_action_at_T_minus_1, td_targets)
    
    # (如果實現了 PER，這裡需要乘以 importance_weights)
    # if importance_weights is not None:
    #     loss = (importance_weights * F.smooth_l1_loss(q_values_for_action_at_T_minus_1, td_targets, reduction='none')).mean()
    # else:
    #     loss = F.smooth_l1_loss(q_values_for_action_at_T_minus_1, td_targets)
            
    if train_steps_count % 1500 == 0: # 每 200 步打印一次損失
        print(f"    [Train Step {train_steps_count}] Loss: {loss.item():.5f}, Epsilon: {epsilon:.3f}")

    # --- 4. 反向傳播和優化 ---
    optimizerB.zero_grad()
    loss.backward()
    # (可選) 梯度裁剪，對於 RNN 訓練可能比較重要
    torch.nn.utils.clip_grad_norm_(modelB.parameters(), max_norm=get_cfg('grad_clip_norm', 1.0))
    optimizerB.step()

    # --- 5. (如果實現了 PER) 更新優先級 ---
    # if sampled_indices is not None and importance_weights is not None:
    #     td_errors_abs = (q_values_for_action_at_T_minus_1 - td_targets).detach().abs().cpu().numpy()
    #     memory.update_priorities(sampled_indices, td_errors_abs) # memory.update_priorities 需要能處理

    train_steps_count += 1
    if train_steps_count % target_update_interval == 0:
        targetB.load_state_dict(modelB.state_dict())
        print(f"    [INFO] Target network updated at train step {train_steps_count}.")


# ------------------- 評估函數 (需要為 RNN 修改) -------------------
def eval_model_vs_opponent(env_instance, model_to_eval, opponent_model, num_episodes):
    model_to_eval.eval() # 確保評估模式
    if opponent_model: opponent_model.eval()

    wins_for_eval_model = 0
    is_eval_rnn = isinstance(model_to_eval, QNetRNN)
    is_opp_rnn = isinstance(opponent_model, QNetRNN) if opponent_model else False

    for _ in range(num_episodes):
        obs_A_single, obs_B_single = env_instance.reset()
        done = False
        
        # 初始化隱藏狀態
        if is_eval_rnn:
            h_eval, c_eval = model_to_eval.init_hidden(1, device)
        if is_opp_rnn:
            h_opp, c_opp = opponent_model.init_hidden(1, device)

        score_A, score_B = 0, 0 # 記錄單局得分

        while not done:
            # Opponent (modelA or pool model) action
            if opponent_model:
                if is_opp_rnn:
                    # 假設 obs_A_single 是 opponent (player A) 的觀測
                    act_opp, (h_opp_next, c_opp_next) = select_action_for_model(opponent_model, obs_A_single, h_opp, c_opp, 0.0, is_eval_mode=True)
                    h_opp, c_opp = h_opp_next, c_opp_next
                else: # Opponent is old QNet
                    act_opp = opponent_model(torch.tensor(obs_A_single, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            else: # No opponent, e.g. for single agent eval if needed (not in this 2P setup)
                act_opp = random.randint(0, env_instance.action_space.nvec[0]-1 if hasattr(env_instance.action_space, 'nvec') else 2)


            # Model to evaluate (modelB) action
            if is_eval_rnn:
                # 假設 obs_B_single 是 model_to_eval (player B) 的觀測
                act_eval, (h_eval_next, c_eval_next) = select_action_for_model(model_to_eval, obs_B_single, h_eval, c_eval, 0.0, is_eval_mode=True)
                h_eval, c_eval = h_eval_next, c_eval_next
            else: # Should not happen if model_to_eval is QNetRNN
                act_eval = model_to_eval(torch.tensor(obs_B_single, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            
            # 環境交互 (假設 Player A 是 opponent, Player B 是 model_to_eval)
            (next_obs_A, next_obs_B), (reward_A, reward_B), done, _ = env_instance.step(act_opp, act_eval)
            
            obs_A_single, obs_B_single = next_obs_A, next_obs_B
            score_A += reward_A
            score_B += reward_B

        if score_B > score_A: # model_to_eval (B) wins
            wins_for_eval_model += 1
            
    model_to_eval.train() # 恢復訓練模式
    return wins_for_eval_model / num_episodes

# ------------------- 載入對手池 (需要處理 RNN 和 非RNN 模型) -------------------
pool_models = []
# 假設舊的 checkpoint 在 'checkpoints/' 目錄，新的 RNN checkpoint 在 `ckpt_dir`
# 您需要一個策略來決定從哪個目錄加載，以及如何區分模型類型
# 這裡簡化：假設對手池主要還是舊的 QNet 模型，或者您有辦法識別 RNN 模型
# Example:
# for fn in os.listdir('checkpoints/'): # 假設舊模型目錄
#     if not fn.endswith('.pth'): continue
#     try:
#         cp2 = torch.load(os.path.join('checkpoints/', fn), map_location=device)
#         st2 = cp2.get('modelB', cp2.get('model', None))
#         if st2 is None: continue
#         m = QNet(input_dim=7, output_dim=3).to(device) # 創建舊 QNet 實例
#         m.load_state_dict(st2, strict=False)
#         m.eval()
#         pool_models.append(m)
#     except Exception as e:
#         print(f"Warning: Could not load pool model {fn}: {e}")
# print(f"[INFO] Loaded {len(pool_models)} (old QNet) pool models.")
# 如果也想從新的 ckpt_dir 加載 RNN 對手：
for fn in os.listdir(ckpt_dir): # RNN 模型目錄
    if not fn.endswith('.pth') or "fault" in fn : continue # 跳過 fault 模型
    try:
        cp_rnn = torch.load(os.path.join(ckpt_dir, fn), map_location=device)
        st_rnn = cp_rnn.get('modelB_state', cp_rnn.get('modelA_state', cp_rnn.get('model'))) # 假設保存 RNN 狀態的鍵
        if st_rnn is None: continue
        m_rnn = create_qnet_rnn_model() # 創建 QNetRNN 實例
        m_rnn.load_state_dict(st_rnn)
        m_rnn.eval()
        pool_models.append(m_rnn)
        print(f"[INFO] Loaded RNN pool model: {fn}")
    except Exception as e:
        print(f"Warning: Could not load RNN pool model {fn}: {e}")
if not pool_models:
    print("[WARNING] Opponent pool is empty! ModelB will only train against ModelA.")


# ------------------- Self‑play 主循環 (大部分邏輯與原文件相似，但調用已修改的函數) ---
done_generations   = 0
current_generation = 0 # 從 0 開始或從 checkpoint 恢復

def reset_model_b_for_new_attempt():
    """重置 modelB 為初始狀態 (通常是 modelA 的一個副本或隨機初始化)"""
    global modelB, optimizerB, memory, epsilon, targetB, train_steps_count
    print("[INFO] Resetting modelB for a new attempt in current generation.")
    
    # modelB 從 old_state_for_reset (通常是 modelA 的初始狀態或一個乾淨的隨機RNN) 恢復
    # modelB = create_qnet_rnn_model() # 確保是新的實例以避免 Optimizer 狀態問題
    # modelB.load_state_dict(old_state_for_reset) # 載入狀態
    # 為了從完全隨機開始，或者如果 old_state_for_reset 就是隨機的，可以這樣：
    modelB = create_qnet_rnn_model() # 重新創建以確保是乾淨的隨機模型
    if init_model_path and os.path.exists(init_model_path) and 'modelB_state' in torch.load(init_model_path, map_location=device):
         # 如果初始 checkpoint 中有 modelB 的特定狀態，則使用它
        checkpoint = torch.load(init_model_path, map_location=device)
        modelB_init_state = checkpoint.get('modelB_state', checkpoint.get('model'))
        modelB.load_state_dict(modelB_init_state)
        print("[INFO] modelB reset to state from init_model_path.")
    else: # 否則，讓它從 modelA 的當前狀態複製 (如果 modelA 是 RNN) 或重新隨機初始化
        if isinstance(modelA, QNetRNN): # 如果 modelA 也是 RNN
            modelB.load_state_dict(modelA.state_dict()) # 和 modelA 一樣開始
            print("[INFO] modelB reset to current modelA's state.")
        else: # modelA 是舊 QNet，modelB 是新 RNN，則 modelB 只能隨機開始或從其自身初始存檔開始
            # modelB 已經是 create_qnet_rnn_model() 創建的隨機模型了
            print("[INFO] modelB reset to a new random QNetRNN state (as modelA is not RNN or no RNN init path for B).")


    optimizerB = optim.Adam(modelB.parameters(), lr=lr) # 為新的 modelB 實例創建新的優化器
    
    # Replay buffer 通常不清空，除非策略如此
    # memory = SequenceReplayBuffer(memory_size, trace_length) # 如果需要清空
    
    epsilon = 1.0 # 重置探索率 (或者根據策略調整)
    targetB = create_qnet_rnn_model() # 新的 target network
    targetB.load_state_dict(modelB.state_dict())
    targetB.eval()
    train_steps_count = 0
    # frame_idx_for_beta = 0 # if PER

# --- 主循環 ---
while done_generations < max_generations:
    current_generation += 1
    print(f"\n=== RNN Training: Generation {current_generation}/{max_generations} ===")
    
    # 在每個新 generation 開始時，modelB 應該是從 modelA 的狀態開始 (如果 modelA 也是 RNN)
    # 或者從一個乾淨的狀態開始，這取決於 self-play 的具體策略
    # 這裡假設 modelB 在新 generation 開始時，繼承上一代成功的 modelA
    if current_generation > 1 : # 第一代 modelB 已經在上面初始化好了
        modelB = create_qnet_rnn_model()
        modelB.load_state_dict(modelA.state_dict()) # B 從 A 的狀態開始學習
        optimizerB = optim.Adam(modelB.parameters(), lr=lr)
        targetB = create_qnet_rnn_model()
        targetB.load_state_dict(modelB.state_dict())
        targetB.eval()
        epsilon = get_cfg('initial_epsilon_per_generation', 1.0) # 每代開始時的 epsilon
        print(f"[INFO] New generation: modelB starts from modelA's state. Epsilon reset to {epsilon}.")


    generation_successful = False
    for i_try in range(1, max_retries + 1):
        print(f"  [Gen {current_generation}] Attempt {i_try}/{max_retries}")
        
        # 確保 modelB 處於訓練模式
        modelB.train()

        for i_episode in range(episodes_per_generation):
            global_episode_count += 1
            
            use_pool_opponent = pool_models and random.random() < opponent_pool_ratio
            opponent_agent = random.choice(pool_models) if use_pool_opponent else modelA
            is_opp_agent_rnn = isinstance(opponent_agent, QNetRNN)

            obs_A_curr, obs_B_curr = env.reset()
            done_episode = False
            episode_reward_b = 0

            # 初始化隱藏狀態
            h_B, c_B = modelB.init_hidden(1, device)
            if is_opp_agent_rnn:
                h_opp, c_opp = opponent_agent.init_hidden(1, device)
            else: # 為非 RNN 對手創建佔位符，雖然不會被使用
                h_opp, c_opp = None, None 

            current_trace = [] # 用於收集一個訓練 trace

            for step_in_episode in range(cfg['env'].get('max_episode_steps', 1000)): # 避免無限循環
                # 對手動作
                if is_opp_agent_rnn:
                    act_A, (h_opp_next, c_opp_next) = select_action_for_model(opponent_agent, obs_A_curr, h_opp, c_opp, 0.0, is_eval_mode=True)
                    h_opp, c_opp = h_opp_next, c_opp_next
                else: # 非 RNN 對手
                    with torch.no_grad():
                         act_A = opponent_agent(torch.tensor(obs_A_curr, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
                
                # modelB 動作
                act_B, (h_B_next, c_B_next) = select_action_for_model(modelB, obs_B_curr, h_B, c_B, epsilon)
                
                (next_obs_A, next_obs_B), (reward_A, reward_B), done_episode, _ = env.step(act_A, act_B)
                
                episode_reward_b += reward_B
                
                # 存儲 transition (之後會用於構建序列)
                # 注意：這裡存儲的是單幀，SequenceReplayBuffer 的 push_step 會處理
                memory.push_step(obs_B_curr, act_B, reward_B, next_obs_B, done_episode, (h_B, c_B), (None,None)) # 暫不存儲 target 隱藏狀態

                obs_A_curr, obs_B_curr = next_obs_A, next_obs_B
                h_B, c_B = h_B_next, c_B_next # 更新隱藏狀態

                # 訓練 (在收集到足夠數據後)
                if len(memory) > batch_size * get_cfg('min_episodes_for_training_start', 5): # 例如，至少有5個 batch 的 episodes
                    train_step_rnn() # 實際的 RNN 訓練邏輯待實現

                if done_episode:
                    break
            
            # Episode 結束後的處理
            win_flag = 1 if episode_reward_b > 0 else 0
            (winP_deque if use_pool_opponent else winA_deque).append(win_flag)
            reward_history.append(episode_reward_b)

            if global_episode_count % win_rate_interval == 0 and (len(winA_deque) > 0 or len(winP_deque) > 0):
                time_now = time.time()
                interval_duration = time_now - interval_start_time
                total_duration_minutes = (time_now - start_time) / 60
                avg_win_A = sum(winA_deque)/len(winA_deque) if len(winA_deque) > 0 else -1
                avg_win_P = sum(winP_deque)/len(winP_deque) if len(winP_deque) > 0 else -1
                print(f"[Ep {global_episode_count}] WinRate (vs A):{avg_win_A:.2f} (vs P):{avg_win_P:.2f} | "
                      f"Eps: {epsilon:.3f} | Last Reward B: {episode_reward_b:.1f} | "
                      f"Interval:{interval_duration:.1f}s | Total:{total_duration_minutes:.1f}min")
                interval_start_time = time_now
            
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # -- Generation attempt 結束，進行評估 --
        print(f"  [Gen {current_generation}, Try {i_try}] Evaluating modelB...")
        # 確保 modelA 和 modelB 在評估時使用正確的模式
        modelB.eval() # 評估 modelB
        # modelA 已經是 eval 模式

        win_rate_vs_A = eval_model_vs_opponent(env, modelB, modelA, eval_episodes)
        # 對手池評估
        if pool_models:
            win_rate_vs_Pool = eval_model_vs_opponent(env, modelB, None, eval_episodes) # TODO: eval_model_vs_opponent 需要能處理對手池
            # 簡化版：隨機選一個 pool model 評估，或者實現一個循環評估所有 pool model
            # temp_pool_opponent = random.choice(pool_models)
            # win_rate_vs_Pool = eval_model_vs_opponent(env, modelB, temp_pool_opponent, eval_episodes)
            # 更準確的是實現一個 `eval_vs_pool_rnn` 函數
            # 臨時使用一個簡單的實現
            if len(pool_models) > 0 :
                total_pool_wins = 0
                num_pool_eval_eps_each = max(1, eval_episodes // len(pool_models))
                for pool_opp in pool_models:
                    total_pool_wins += eval_model_vs_opponent(env, modelB, pool_opp, num_pool_eval_eps_each) * num_pool_eval_eps_each
                win_rate_vs_Pool = total_pool_wins / (len(pool_models) * num_pool_eval_eps_each)
            else:
                win_rate_vs_Pool = 1.0 # 如果池為空，則認為通過
        else:
            win_rate_vs_Pool = 1.0 # 如果池為空，則認為通過 (符合原邏輯)
        
        modelB.train() # 恢復 modelB 到訓練模式

        print(f"  [Gen {current_generation}, Try {i_try}] Eval Results: vs A:{win_rate_vs_A:.2f}, vs Pool:{win_rate_vs_Pool:.2f}, Eps:{epsilon:.3f}")

        if win_rate_vs_A >= curr_win_threshold and win_rate_vs_Pool >= pool_win_threshold:
            print(f"  SUCCESS! ModelB passed thresholds in Gen {current_generation}, Try {i_try}.")
            # 升級 modelA
            modelA.load_state_dict(modelB.state_dict())
            modelA.eval() # 確保新的 modelA 也是評估模式
            # for p in modelA.parameters(): p.requires_grad = False # 這一行其實不需要，因為 eval() 會處理

            # 保存成功的模型 (modelB 的狀態，現在也是 modelA 的狀態)
            # 使用新的前綴和 generation 編號
            save_filename = f"{rnn_model_id_prefix}{current_generation}.pth"
            torch.save({
                'modelA_state': modelA.state_dict(), # 保存為 modelA 的狀態，因為它已經被更新
                'modelB_state': modelB.state_dict(), # 也保存 modelB 的狀態 (此刻與A相同)
                'optimizer_B_state': optimizerB.state_dict(), # 可能不需要，因為下一代 B 會重新創建優化器
                'epsilon': epsilon,
                'episode': global_episode_count,
                'generation': current_generation
            }, os.path.join(ckpt_dir, save_filename))
            print(f"  [Saved] Checkpoint: {os.path.join(ckpt_dir, save_filename)}")
            
            # 如果希望將成功的模型也加入到 pool_models (運行時的)
            newly_promoted_model = create_qnet_rnn_model()
            newly_promoted_model.load_state_dict(modelA.state_dict())
            newly_promoted_model.eval()
            pool_models.append(newly_promoted_model) # 動態添加到當前運行的對手池
            print(f"  Added {save_filename} to the runtime opponent pool (now {len(pool_models)} models).")

            done_generations += 1
            generation_successful = True
            break # 跳出當前 generation 的 retries 循環
        else:
            print(f"  ModelB did not meet thresholds in Gen {current_generation}, Try {i_try}. Continuing...")
            if i_try < max_retries:
                 # 可以在這裡重置 modelB 的一部分狀態，或者調整 epsilon 等，而不是完全重置
                 # 例如，稍微增加 epsilon
                 # epsilon = min(1.0, epsilon * 1.5) # 臨時增加探索
                 # print(f"    Increased epsilon to {epsilon} for next try.")
                 pass # 繼續下一次嘗試，modelB 狀態保持
            # 如果是最後一次嘗試失敗，則在循環外處理 fault

        # --- try 循環結束 ---

        if not generation_successful and i_try == max_retries:
            print(f"  FAILURE! ModelB FAILED to pass thresholds after {max_retries} tries in Gen {current_generation}.")
            fault_filename = f"{rnn_model_id_prefix}{current_generation}_fault.pth"
            torch.save({
                'modelB_state': modelB.state_dict(), # 保存失敗的 modelB
                'optimizer_B_state': optimizerB.state_dict(),
                'epsilon': epsilon,
                'episode': global_episode_count,
                'generation': current_generation,
                'modelA_state': modelA.state_dict() # 同時保存當時的 modelA 狀態
            }, os.path.join(ckpt_dir, fault_filename))
            print(f"  [Fault Saved] Checkpoint: {os.path.join(ckpt_dir, fault_filename)}")
            
            # 重置 modelB 以準備下一個 generation (或者如果這是 self-play 的結束條件)
            # 根據原邏輯，即使 fault，也算完成了一個 generation 的嘗試
            reset_model_b_for_new_attempt() # 或者決定是否真的要重置
            done_generations += 1 # 原邏輯是 fault 也算完成一代
            break # 跳出 try 循環，進入下一個 generation

# --- generation 循環結束 ---

env.close()

# ------------------- 繪圖 (與原文件類似) -------------------
if reward_history:
    window = 50
    if len(reward_history) >= window:
        smooth_rewards = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    else:
        smooth_rewards = np.array(reward_history) # 如果數據不足，不平滑或用原始數據

    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, alpha=0.4, label='Raw Reward B (RNN)')
    if len(reward_history) >= window:
        plt.plot(range(window-1, len(reward_history)), smooth_rewards, color='red', label=f'Smoothed Reward (window {window})')
    else:
        plt.plot(smooth_rewards, color='red', label='Reward (Raw - less than window size)')
    plt.legend()
    plt.title(f"RNN Self-play Training Rewards ({rnn_model_id_prefix})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Model B)")
    plt.grid(True)
    plot_filename = os.path.join(plot_dir, f"training_rnn_rewards_gen{current_generation}.png")
    plt.savefig(plot_filename)
    print(f"Training finished! Plot saved to {plot_filename}")
else:
    print("Training finished! No reward history to plot.")