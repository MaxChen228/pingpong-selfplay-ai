import os
import yaml
import random
import numpy as np
import torch
# 限制 BLAS/OpenMP 线程数 & PyTorch 线程数
os.environ['OMP_NUM_THREADS']    = '8'
os.environ['MKL_NUM_THREADS']    = '8'
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import copy
import time

from envs.my_pong_env_2p import PongEnv2P
from models.qnet       import QNet  # NoisyLinear 已在 models/qnet.py 中定义

# ------------------- 读取配置 -------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
def get_cfg(key, default=None):
    return cfg['training'].get(key, default)

# 训练超参
max_generations         = get_cfg('max_generations')
episodes_per_generation = get_cfg('episodes_per_generation')
eval_episodes           = get_cfg('eval_episodes')
curr_win_threshold      = get_cfg('curr_win_threshold')
pool_win_threshold      = get_cfg('pool_win_threshold')
lr                      = get_cfg('lr')
gamma                   = get_cfg('gamma')
batch_size              = get_cfg('batch_size')
memory_size             = get_cfg('memory_size')
epsilon_decay           = get_cfg('epsilon_decay')
min_epsilon             = get_cfg('min_epsilon')
init_model_path         = get_cfg('init_model_path')
model_id                = get_cfg('model_id')
opponent_pool_ratio     = get_cfg('opponent_pool_ratio')
target_update_interval  = get_cfg('target_update_interval')
win_rate_interval       = get_cfg('win_rate_interval')
max_retries             = get_cfg('max_retries_for_generation')

ckpt_dir = os.path.dirname(init_model_path) or '.'

# ------------------- PER 回放池 -------------------
class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6):
        self.cap    = capacity
        self.alpha  = alpha
        self.buffer = []
        self.pos    = 0
        self.prios  = np.zeros((capacity,), dtype=np.float32)
    def push(self, trans):
        max_p = self.prios.max() if self.buffer else 1.0
        if len(self.buffer) < self.cap:
            self.buffer.append(trans)
        else:
            self.buffer[self.pos] = trans
        self.prios[self.pos] = max_p
        self.pos = (self.pos + 1) % self.cap
    def sample(self, bs, beta=0.4):
        if not self.buffer: return [], [], None
        pr   = self.prios if len(self.buffer)==self.cap else self.prios[:self.pos]
        probs= pr**self.alpha; probs/=probs.sum()
        idxs = np.random.choice(len(self.buffer), bs, p=probs)
        batch= [self.buffer[i] for i in idxs]
        total= len(self.buffer)
        weights = (total * probs[idxs])**(-beta)
        weights /= weights.max()
        return batch, idxs, torch.tensor(weights, device=device)
    def update_priorities(self, idxs, errors):
        for i,e in zip(idxs, errors):
            self.prios[i] = abs(e) + 1e-6

# ------------------- 环境 & 设备 -------------------
env    = PongEnv2P(**cfg['env'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    return QNet(input_dim=7, output_dim=3).to(device)

# ------------------- 加载旧 checkpoint -------------------
base_cp   = torch.load(init_model_path, map_location=device)
old_state = base_cp.get('modelB', base_cp.get('model'))

# ------------------- 初始化 modelA & modelB -------------------
modelA = create_model()
modelA.load_state_dict(old_state, strict=False)
for p in modelA.parameters(): p.requires_grad = False

modelB = create_model()
modelB.load_state_dict(old_state, strict=False)
# 冻结 feature 层，只训练 Noisy Head
for p in modelB.features.parameters(): p.requires_grad = False

# ------------------- 目标网络 & 优化器 -------------------
targetB   = copy.deepcopy(modelB); targetB.eval()
optimizerB= optim.Adam(
    list(modelB.fc_V.parameters()) + list(modelB.fc_A.parameters()),
    lr=lr
)

epsilon              = base_cp.get('epsilon', min_epsilon)
global_episode_count = base_cp.get('episode', 0)
print(f"[INFO] Loaded init_model from {init_model_path}, eps={epsilon}, episode={global_episode_count}")

# ------------------- 初始化 PER、记录与计时 -------------------
memory         = PrioritizedReplay(memory_size, alpha=0.6)
beta_start     = 0.4
beta_frames    = 100000
frame_idx      = 0
reward_history = []
winA           = deque(maxlen=win_rate_interval)
winP           = deque(maxlen=win_rate_interval)
train_steps    = 0

start_time     = time.time()
interval_start = start_time

# ------------------- 动作选择 & 训练步 -------------------
def select_action_B(obs):
    modelB.reset_noise()   # 刷新噪声
    if random.random() < epsilon:
        return random.randint(0, 2)
    with torch.no_grad():
        return modelB(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))\
                     .argmax(1).item()

def train_step():
    global train_steps, frame_idx
    if len(memory.buffer) < batch_size:
        return
    frame_idx += 1
    beta = min(1.0, beta_start + frame_idx*(1.0-beta_start)/beta_frames)
    batch, idxs, iw = memory.sample(batch_size, beta)
    if not batch: return

    # 一次性刷新噪声
    modelB.reset_noise()
    targetB.reset_noise()

    states, actions, rewards_, next_states, dones = zip(*batch)
    s  = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    a  = torch.tensor(actions,      dtype=torch.int64,   device=device)
    r  = torch.tensor(rewards_,     dtype=torch.float32, device=device)
    ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    d  = torch.tensor(dones,        dtype=torch.bool,    device=device)

    q_vals = modelB(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        na  = modelB(ns).argmax(1, keepdim=True)
        nq  = targetB(ns).gather(1, na).squeeze(1)
    targets = r + gamma * nq * (~d)

    loss = (iw * (q_vals - targets).pow(2)).mean()
    optimizerB.zero_grad()
    loss.backward()
    optimizerB.step()

    errors = (q_vals - targets).detach().abs().cpu().numpy()
    memory.update_priorities(idxs, errors)

    train_steps += 1
    if train_steps % target_update_interval == 0:
        targetB.load_state_dict(modelB.state_dict())

# ------------------- 评估函数 -------------------
def eval_vs_model(env, A, B, episodes):
    wins = 0
    for _ in range(episodes):
        oA, oB = env.reset(); done=False
        while not done:
            aA = A(torch.tensor(oA, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            aB = B(torch.tensor(oB, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            (nA, nB), (rA, rB), done, _ = env.step(aA, aB)
            oA, oB = nA, nB
        if rB > rA: wins += 1
    return wins / episodes

def eval_vs_pool(env, B, pool, episodes):
    if not pool:
        return 1.0
    wins = 0
    for _ in range(episodes):
        opp = random.choice(pool)
        oA, oB = env.reset(); done=False
        while not done:
            aA = opp(torch.tensor(oA, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            aB = B(torch.tensor(oB, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            (nA, nB), (rA, rB), done, _ = env.step(aA, aB)
            oA, oB = nA, nB
        if rB > rA: wins += 1
    return wins / episodes

# ------------------- 载入对手池 -------------------
pool_models = []
for fn in os.listdir(ckpt_dir):
    if not fn.endswith('.pth'): continue
    cp2 = torch.load(os.path.join(ckpt_dir, fn), map_location=device)
    st2 = cp2.get('modelB', cp2.get('model', None))
    if st2 is None: continue
    m = create_model(); m.load_state_dict(st2, strict=False); m.eval()
    pool_models.append(m)
print(f"[INFO] Loaded {len(pool_models)} pool models.")

# ------------------- Self‑play 主循环 -------------------
done_generations   = 0
current_generation = 0

def reset_B():
    global modelB, optimizerB, memory, epsilon, targetB, train_steps, frame_idx
    modelB = create_model(); modelB.load_state_dict(old_state, strict=False)
    for p in modelB.features.parameters(): p.requires_grad=False
    optimizerB = optim.Adam(
        list(modelB.fc_V.parameters()) + list(modelB.fc_A.parameters()),
        lr=lr
    )
    memory       = PrioritizedReplay(memory_size, alpha=0.6)
    epsilon      = 1.0
    targetB      = copy.deepcopy(modelB); targetB.eval()
    train_steps  = 0; frame_idx = 0

while done_generations < max_generations:
    current_generation += 1
    print(f"\n=== Generation {current_generation} ===")
    tries = 0
    while True:
        tries += 1
        print(f"  [Gen {current_generation}] try {tries}/{max_retries}")
        for _ in range(episodes_per_generation):
            global_episode_count += 1
            use_pool = pool_models and random.random() < opponent_pool_ratio
            opp = random.choice(pool_models) if use_pool else modelA

            oA, oB = env.reset(); done=False; ep_reward=0
            while not done:
                aA = opp(torch.tensor(oA, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
                aB = select_action_B(oB)
                (nA,nB),(rA,rB),done,_ = env.step(aA,aB)
                memory.push((oB, aB, rB, nB, done))
                train_step()
                oA, oB = nA, nB; ep_reward += rB

            win = 1 if ep_reward>0 else 0
            (winP if use_pool else winA).append(win)
            reward_history.append(ep_reward)

            if global_episode_count % win_rate_interval == 0:
                now = time.time()
                interval = now - interval_start
                total    = now - start_time
                avgA = sum(winA)/len(winA)
                avgP = sum(winP)/len(winP)
                print(f"[Ep {global_episode_count}] vs A:{avgA:.2f}, vs Pool:{avgP:.2f}, "
                      f"interval:{interval:.1f}s, total:{total/60:.1f}min")
                interval_start = now

            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        wA = eval_vs_model(env, modelA, modelB, eval_episodes)
        wP = eval_vs_pool(env, modelB, pool_models, eval_episodes)
        print(f"[Gen {current_generation}] vs A:{wA:.2f}, vs Pool:{wP:.2f}, eps={epsilon:.3f}")

        if wA >= curr_win_threshold and wP >= pool_win_threshold:
            print(f"升級! generation {current_generation} done.")
            modelA.load_state_dict(modelB.state_dict())
            for p in modelA.parameters(): p.requires_grad=False
            fn = f"model{model_id}-{current_generation}.pth"
            torch.save({
                'modelB'   : modelB.state_dict(),
                'optimizer': optimizerB.state_dict(),
                'epsilon'  : epsilon,
                'episode'  : global_episode_count,
                'modelA'   : modelA.state_dict()
            }, os.path.join(ckpt_dir, fn))
            print(f"[Saved] {fn}")
            done_generations += 1
            break
        else:
            if tries >= max_retries:
                fn = f"model{model_id}-{current_generation}_fault.pth"
                torch.save({
                    'modelB'   : modelB.state_dict(),
                    'optimizer': optimizerB.state_dict(),
                    'epsilon'  : epsilon,
                    'episode'  : global_episode_count,
                    'modelA'   : modelA.state_dict()
                }, os.path.join(ckpt_dir, fn))
                print(f"[Fault] {fn}")
                reset_B()
                done_generations += 1
                break
            else:
                print("未達标，继续尝试…")

env.close()

# ------------------- 绘图 -------------------
window = 50
smooth = np.convolve(reward_history, np.ones(window)/window, mode='valid')
plt.figure(figsize=(10,4))
plt.plot(reward_history, alpha=0.3, label='Reward B')
plt.plot(range(window-1, len(reward_history)), smooth, label='Smoothed')
plt.legend()
plt.title("Self-play with NoisyNet & PER")
plt.xlabel("Episode")
plt.ylabel("Reward B")
os.makedirs("plot", exist_ok=True)
plt.savefig("plot/training_iterative_rewards.png")
print("Done! See plot/training_iterative_rewards.png")
