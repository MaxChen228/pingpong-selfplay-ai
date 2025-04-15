import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time

from envs.my_pong_env_2p import PongEnv2P
from models.qnet import QNet

# 1) 讀取 config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

max_generations = cfg['training']['max_generations']
episodes_per_generation = cfg['training']['episodes_per_generation']  # 每次測試前的預設訓練場數
eval_episodes = cfg['training']['eval_episodes']
win_threshold = cfg['training']['win_threshold']

lr = cfg['training']['lr']
gamma = cfg['training']['gamma']
batch_size = cfg['training']['batch_size']
memory_size= cfg['training']['memory_size']
epsilon_decay= cfg['training']['epsilon_decay']
min_epsilon= cfg['training']['min_epsilon']

init_model_path = cfg['training']['init_model_path']

# 2) 建立環境
#   移除舊的 "ball_speed=cfg['env']['ball_speed']"，改為帶入 ball_speed_range / spin_range / ball_angle_intervals
#   並將 speed_scale_every, speed_increment 帶入
env = PongEnv2P(
    render_size  = cfg['env']['render_size'],
    paddle_width = cfg['env']['paddle_width'],
    paddle_speed = cfg['env']['paddle_speed'],
    max_score    = cfg['env']['max_score'],
    enable_render= False,

    enable_spin       = cfg['env']['enable_spin'],
    magnus_factor     = cfg['env']['magnus_factor'],
    restitution       = cfg['env']['restitution'],
    friction          = cfg['env']['friction'],
    ball_mass         = cfg['env']['ball_mass'],
    world_ball_radius = cfg['env']['world_ball_radius'],

    ball_angle_intervals = cfg['env']['ball_angle_intervals'],
    ball_speed_range  = tuple(cfg['env']['ball_speed_range']),
    spin_range        = tuple(cfg['env']['spin_range']),

    speed_scale_every = cfg['env']['speed_scale_every'],
    speed_increment   = cfg['env']['speed_increment']
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) 建立 QNet => 7維輸入
def create_model():
    return QNet(input_dim=7, output_dim=3).to(device)

modelA = create_model()
modelB = create_model()
optimizerB = optim.Adam(modelB.parameters(), lr=lr)

# 4) 載入 init_model_path
if os.path.exists(init_model_path):
    checkpoint = torch.load(init_model_path, map_location=device)
    if 'modelA' in checkpoint and 'modelB' in checkpoint:
        modelA.load_state_dict(checkpoint['modelA'])
        modelB.load_state_dict(checkpoint['modelB'])
    else:
        # 兼容只存 'model'
        modelA.load_state_dict(checkpoint['model'])
        modelB.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        optimizerB.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint.get('epsilon', 1.0)
    start_episode = checkpoint.get('episode', 0)
    print(f"[INFO] Loaded init_model from {init_model_path}, eps={epsilon}, episode={start_episode}")
else:
    raise FileNotFoundError(f"init_model_path: {init_model_path} not found!")

# 5) A為固定對手 => 不參與更新
for p in modelA.parameters():
    p.requires_grad = False

memory = deque(maxlen=memory_size)
epsilon = 1.0 if 'epsilon' not in locals() else epsilon
reward_history = []
global_episode_count = 0

def select_action_B(obs):
    # B 訓練階段 => ε-greedy
    if random.random() < epsilon:
        return random.randint(0,2)
    else:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q = modelB(obs_t)
            return q.argmax(dim=1).item()

def select_action_A(obs):
    # A 固定 => 無探索
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q = modelA(obs_t)
        return q.argmax(dim=1).item()

def train_step():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards_, next_states, dones = zip(*batch)

    states     = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions    = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_   = torch.tensor(rewards_, dtype=torch.float32, device=device)
    next_states= torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones      = torch.tensor(dones, dtype=torch.bool, device=device)

    q_values   = modelB(states)
    q_val      = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = modelB(next_states).max(1)[0]
    expected_q = rewards_ + gamma * next_q * (~dones)

    loss = nn.MSELoss()(q_val, expected_q)
    optimizerB.zero_grad()
    loss.backward()
    optimizerB.step()

# 評估 => A,B 用 argmax
def select_action_eval(obs, model_):
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model_(obs_t)
        return q_values.argmax(dim=1).item()

def evaluate_win_rate(env, modelA, modelB, episodes=10):
    wins = 0
    for _ in range(episodes):
        obsA, obsB = env.reset()
        done = False
        while not done:
            actA = select_action_eval(obsA, modelA)
            actB = select_action_eval(obsB, modelB)
            (nextA, nextB), (rA, rB), done, _ = env.step(actA, actB)
            obsA, obsB = nextA, nextB
            if done and rB > rA:
                wins += 1
    return wins / episodes

done_generations = 0
current_generation = 0

# 多世代迴圈
while done_generations < max_generations:
    current_generation += 1
    print(f"\n=== Generation {current_generation} ===")

    while True:
        # (1) 先做 episodes_per_generation 場訓練
        for ep in range(episodes_per_generation):
            global_episode_count += 1
            obsA, obsB = env.reset()
            done=False
            ep_rewardB=0
            while not done:
                actA = select_action_A(obsA)
                actB = select_action_B(obsB)
                (nextA, nextB), (rA, rB), done, _ = env.step(actA, actB)

                memory.append((obsB, actB, rB, nextB, done))
                train_step()

                obsA, obsB = nextA, nextB
                ep_rewardB += rB

            reward_history.append(ep_rewardB)
            # epsilon decay
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # (2) 評估
        wr = evaluate_win_rate(env, modelA, modelB, episodes=eval_episodes)
        print(f"[Gen {current_generation}] Evaluate B => WinRate={wr:.2f}, eps={epsilon:.3f}")

        # 若 B 達標 => 升級 => 進下一代
        if wr >= win_threshold:
            print(f"B surpasses A! => generation {current_generation} done.")
            modelA.load_state_dict(modelB.state_dict())
            for p in modelA.parameters():
                p.requires_grad = False

            # 儲存 => 'model' 放 B
            ckpt_path = f"checkpoints/model2_gen{current_generation}.pth"
            torch.save({
                'model': modelB.state_dict(),
                'optimizer': optimizerB.state_dict(),
                'epsilon': epsilon,
                'episode': global_episode_count,
                'modelA': modelA.state_dict(),
                'modelB': modelB.state_dict()
            }, ckpt_path)
            print(f"[Saved] {ckpt_path}")

            done_generations += 1
            break
        else:
            # 沒達標 => 留在同一代繼續訓練
            print("B not surpass A yet. Keep training in the same generation...")

    if done_generations >= max_generations:
        print("Reached max_generations => stop.")
        break

env.close()

# 畫圖
window=50
smooth = np.convolve(reward_history, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,4))
plt.plot(reward_history, alpha=0.3, label='Reward B per Episode')
plt.plot(range(window-1, len(reward_history)), smooth, label='Smoothed')
plt.legend()
plt.title("Reward of B during Iterative Self-Play (Stay in Gen if Fail)")
plt.xlabel("Episode")
plt.ylabel("Reward B")
os.makedirs("plot", exist_ok=True)
plt.savefig("plot/training_iterative_rewards.png")
print("Done! See 'plot/training_iterative_rewards.png'")
