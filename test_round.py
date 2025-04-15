#!/usr/bin/env python3
import os
import yaml
import torch
import random
import numpy as np

from envs.my_pong_env_2p import PongEnv2P
from models.qnet import QNet

def load_model(model_path, device):
    """
    載入 .pth 檔案，回傳一個 QNet 實例
    預期檔案內含 {'model': state_dict, ...}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 假設 7維輸入, 3動作
    model_ = QNet(input_dim=7, output_dim=3).to(device)
    if 'model' in checkpoint:
        model_.load_state_dict(checkpoint['model'])
    else:
        raise KeyError("Checkpoint missing 'model' key, found keys: " + str(checkpoint.keys()))
    model_.eval()
    return model_

def select_action_eval(obs, model_, device):
    """
    不帶探索 => 直接 argmax Q
    """
    import torch
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model_(obs_t)
        action = q_values.argmax(dim=1).item()
    return action

def main():
    # 1) 讀取 config.yaml，不要渲染
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    # 強制不渲染
    env_cfg["enable_render"] = False

    # 2) 準備要參賽的模型路徑
    # 例如可以放多個檔案:
    model_paths = [
        "checkpoints/model_gen0.pth",
        "checkpoints/model_gen3.pth",
        "checkpoints/model2_gen0.pth",
        "checkpoints/model2_gen3.pth",
        "checkpoints/model2_gen6.pth"
        # 你可以再加 ...
    ]
    # 參賽者清單
    participants = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mpath in model_paths:
        # 讀取 & 存成 (model_name, net物件, 累積勝場, 累積敗場)
        net = load_model(mpath, device)
        participants.append({
            "name": os.path.basename(mpath),
            "model": net,
            "win": 0,
            "lose": 0
        })

    # 3) 建立環境 (雙人, 不渲染)
    env = PongEnv2P(
        render_size  = env_cfg["render_size"],
        paddle_width = env_cfg["paddle_width"],
        paddle_speed = env_cfg["paddle_speed"],
        max_score    = env_cfg["max_score"],
        enable_render= env_cfg["enable_render"],

        enable_spin       = env_cfg["enable_spin"],
        magnus_factor     = env_cfg["magnus_factor"],
        restitution       = env_cfg["restitution"],
        friction          = env_cfg["friction"],
        ball_mass         = env_cfg["ball_mass"],
        world_ball_radius = env_cfg["world_ball_radius"],

        ball_angle_intervals = env_cfg["ball_angle_intervals"],
        ball_speed_range  = tuple(env_cfg["ball_speed_range"]),
        spin_range        = tuple(env_cfg["spin_range"]),

        speed_scale_every = env_cfg["speed_scale_every"],
        speed_increment   = env_cfg["speed_increment"]
    )

    # 4) 小組賽: 每個 vs. 每個
    #   例如跑 n episodesEach, 累計分數
    episodesEach = 50  # 每組對戰要打幾局
    n = len(participants)

    print("[INFO] Round Robin Start!")
    for i in range(n):
        for j in range(i+1, n):
            # i vs j
            nameA = participants[i]["name"]
            netA  = participants[i]["model"]
            nameB = participants[j]["name"]
            netB  = participants[j]["model"]

            # 兩個對打 episodesEach 場
            winA = 0
            winB = 0
            for ep in range(episodesEach):
                obsA, obsB = env.reset()
                done = False
                scoreA = 0
                scoreB = 0
                while not done:
                    actA = select_action_eval(obsA, netA, device)
                    actB = select_action_eval(obsB, netB, device)
                    (nA,nB), (rA, rB), done, _ = env.step(actA, actB)
                    obsA, obsB = nA, nB
                    scoreA += rA
                    scoreB += rB
                # 最後看誰贏
                if scoreA > scoreB:
                    winA += 1
                elif scoreB > scoreA:
                    winB += 1
                # 如果 scoreA=scoreB 就算平手, 這裡先不計

            print(f"{nameA} vs. {nameB}: A_wins={winA}, B_wins={winB} (out of {episodesEach})")

            # 更新累積勝敗
            if winA>winB:
                participants[i]["win"] += 1
                participants[j]["lose"]+= 1
            elif winB>winA:
                participants[j]["win"] += 1
                participants[i]["lose"]+= 1
            # 若平手就都沒加, 但可自行設計

    # 5) 全部對完 => 排名 (依勝>負)
    #   也可用 "勝場最多" or "勝率" 排
    participants.sort(key=lambda p: p["win"], reverse=True)
    print("\n=== Final Ranking ===")
    rank=1
    for p in participants:
        print(f"Rank {rank}: {p['name']} => W={p['win']}, L={p['lose']}")
        rank+=1

    env.close()

if __name__=="__main__":
    main()
