#!/usr/bin/env python3
"""
test_round.py ─ 通用模型循環賽評估器
====================================================================
- 支持 QNet (舊式 fc.* 或新式 NoisyDueling) 和 QNetRNN 模型進行循環對戰。
- 可在 USER_CONFIG 中方便地配置參賽模型列表、路徑、類型及比賽參數。
- 自動進行所有模型配對的比賽，並統計勝率。
- 可選生成勝率條形圖和 H2H (Head-to-Head) 熱力圖。
- 使用 tqdm 顯示比賽進度。
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import time

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # 用於進度條

from envs.my_pong_env_2p import PongEnv2P
from models.qnet import QNet
from models.qnet_rnn import QNetRNN

# ────────────────── 1. 用戶配置區域 (USER_CONFIG) ──────────────────
USER_CONFIG: Dict[str, Any] = {
    "env_config_path": "config.yaml",       # 環境配置文件路徑
    "rnn_model_config_path": "config_rnn.yaml", # RNN 模型架構參數配置文件路徑

    "models_to_compete": [
        # {
        #     "name": "model2-0", # 自定義模型名稱，用於報告和圖表
        #     "path": "checkpoints/model2-0.pth", # 模型 checkpoint 路徑
        #     "type": "QNet",         # 模型類型: "QNet" 或 "QNetRNN"
        # },
        # {
        #     "name": "model3-0",
        #     "path": "checkpoints/model3-0.pth",
        #     "type": "QNet",
        # },
        # {
        #     "name": "model4-0", # 自定義模型名稱，用於報告和圖表
        #     "path": "checkpoints/model4-0.pth", # 模型 checkpoint 路徑
        #     "type": "QNet",         # 模型類型: "QNet" 或 "QNetRNN"
        # },
        # {
        #     "name": "model4-12",
        #     "path": "checkpoints/model4-12.pth", #
        #     "type": "QNet", #
        # },
        {
            "name": "RNN_Gen1",
            "path": "checkpoints_rnn/rnn_agent_1.pth", # 假設這是你的 RNN 模型
            "type": "QNetRNN",
        },
        {
            "name": "RNN_Gen2",
            "path": "checkpoints_rnn/rnn_agent_2.pth", # 假設這是你的 RNN 模型
            "type": "QNetRNN",
        },
        {
            "name": "RNN_Gen3",
            "path": "checkpoints_rnn/rnn_agent_3.pth", # 假設這是你的 RNN 模型
            "type": "QNetRNN",
        },
        {
            "name": "RNN_Gen4",
            "path": "checkpoints_rnn/rnn_agent_4.pth", 
            "type": "QNetRNN", 
        },
        {
            "name": "RNN_Gen5",
            "path": "checkpoints_rnn/rnn_pong_soul_1.pth", 
            "type": "QNetRNN", 
        },
        {
            "name": "RNN_Gen6",
            "path": "checkpoints_rnn/rnn_pong_soul_2.pth", 
            "type": "QNetRNN", 
        },
        # 你可以繼續添加更多模型進行比賽
        # {
        #     "name": "RNN_Gen2_Fault",
        #     "path": "checkpoints_rnn/rnn_pong_soul_2_fault.pth",
        #     "type": "QNetRNN",
        # },

        # --- 新增的硬編碼代理 ---
        {
            "name": "BallFollowerBot",  # 您可以給它取任何名字
            "path": "N/A",             # 不需要模型檔案
            "type": "HardcodedBallFollower" # 新的類型名稱
        },
        # --- 新增結束 ---
    ],
    "episodes_per_match": 100,  # 每對模型之間進行的比賽局數 #
    "output_dir": "results", # 結果 (CSV, 圖表) 的輸出目錄 #
    "generate_plots": True,     # 是否生成勝率圖和 H2H 熱力圖 #
    "verbose_match_progress": False, # 是否在每局結束後打印詳細進度 (如果局數很多，可以設為 False) #
}
# ────────────────── (用戶配置區域結束) ──────────────────


# ────────────────── 2. 通用模型載入函數 ──────────────────
def load_model_universal(model_info: Dict[str, str],
                         rnn_arch_config: Dict[str, Any],
                         device: torch.device) -> Union[nn.Module, str, None]: # 修改返回類型提示
    model_path_str = model_info["path"]
    model_type = model_info["type"]
    model_name = model_info.get("name", Path(model_path_str).stem) # 用於日誌

    # --- 新增：處理硬編碼代理 ---
    if model_type == "HardcodedBallFollower":
        print(f"  識別到硬編碼代理: {model_name} (類型: {model_type})")
        return "HardcodedAgent" # 返回一個描述性字串或 None
    # --- 新增結束 ---

    path = Path(model_path_str)
    if not path.exists():
        raise FileNotFoundError(f"模型文件 '{model_name}' 未找到於: {path}")

    ckpt = torch.load(path, map_location=device)
    
    state_dict_keys_to_try = ["modelB_state", "modelA_state", "modelB", "modelA", "model", "state_dict"]
    state_dict = None
    for key in state_dict_keys_to_try:
        if key in ckpt:
            state_dict = ckpt[key]
            break
    
    if state_dict is None:
        if all(not isinstance(v, dict) for v in ckpt.values()) and any (k.startswith("fc.") or k.startswith("features.") for k in ckpt.keys()):
             state_dict = ckpt
        else:
            raise KeyError(f"在模型 '{model_name}' 的 checkpoint '{path}' 中找不到有效的模型狀態字典。嘗試的鍵名: {state_dict_keys_to_try}。ckpt keys: {list(ckpt.keys())}")

    input_dim = 7
    output_dim = 3

    if model_type == "QNet":
        net = QNet(input_dim=input_dim, output_dim=output_dim).to(device)
        is_new_qnet_format = any(k.startswith(("features.", "fc_V.", "fc_A.")) for k in state_dict)
        if is_new_qnet_format:
            net.load_state_dict(state_dict, strict=True)
        else:
            mapped_state_dict: Dict[str, torch.Tensor] = {}
            for k_ckpt, v_ckpt in state_dict.items():
                if k_ckpt.startswith("fc.0."): mapped_state_dict[k_ckpt.replace("fc.0.", "features.0.")] = v_ckpt
                elif k_ckpt.startswith("fc.2."): mapped_state_dict[k_ckpt.replace("fc.2.", "features.2.")] = v_ckpt
            if "fc.4.weight" in state_dict and "fc.4.bias" in state_dict:
                w4, b4 = state_dict["fc.4.weight"], state_dict["fc.4.bias"]
                mapped_state_dict["fc_A.weight_mu"] = w4; mapped_state_dict["fc_A.bias_mu"] = b4
                mapped_state_dict["fc_V.weight_mu"] = w4.mean(dim=0, keepdim=True); mapped_state_dict["fc_V.bias_mu"] = b4.mean().unsqueeze(0)
            else:
                 print(f"[警告] 模型 '{model_name}' (舊式 QNet): checkpoint '{path}' 缺少 'fc.4.weight' 或 'fc.4.bias'。")
            net.load_state_dict(mapped_state_dict, strict=False)
    elif model_type == "QNetRNN":
        if not rnn_arch_config: # 如果 rnn_arch_config 為空
            print(f"[警告] 模型 '{model_name}' (QNetRNN): RNN 架構參數未提供或配置文件未找到，將使用預設 QNetRNN 參數。")
        net = QNetRNN(
            input_dim=input_dim, output_dim=output_dim,
            feature_dim=rnn_arch_config.get('feature_dim', 128),
            lstm_hidden_dim=rnn_arch_config.get('lstm_hidden_dim', 128),
            lstm_layers=rnn_arch_config.get('lstm_layers', 1),
            head_hidden_dim=rnn_arch_config.get('head_hidden_dim', 128)
        ).to(device)
        net.load_state_dict(state_dict)
    else:
        # 這個 else 理論上不應該被觸發，因為 HardcodedBallFollower 已在前面處理
        raise ValueError(f"不支持的模型類型 '{model_type}' (模型: {model_name})")

    net.eval()
    if hasattr(net, 'reset_noise'):
        net.reset_noise() # 確保 NoisyNet 的噪聲狀態一致 (對於評估，這通常意味著使用 mu 值)
    return net

# ────────────────── 3. 通用動作選擇函數 ──────────────────
def select_action_universal(obs: np.ndarray, model: Union[nn.Module, str, None], model_type: str, # 修改 model 的類型提示
                            hidden_state: Tuple[torch.Tensor, torch.Tensor] | None,
                            device: torch.device
                           ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor] | None]:
    with torch.no_grad():
        if model_type == "QNetRNN":
            assert hidden_state is not None, "QNetRNN 需要 hidden_state"
            assert isinstance(model, QNetRNN), "模型實例與類型 'QNetRNN' 不符" # model 此時應為 nn.Module
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            q_values, next_hidden_state = model(obs_tensor, hidden_state)
            action = int(q_values.argmax(1).item())
            return action, next_hidden_state
        elif model_type == "QNet":
            assert isinstance(model, QNet), "模型實例與類型 'QNet' 不符" # model 此時應為 nn.Module
            # QNet 在 eval 模式下，NoisyLinear 會自動使用 mu 值，不需要手動 reset_noise() 除非有特殊需求
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(obs_tensor)
            action = int(q_values.argmax(1).item())
            return action, None
        # --- 新增：處理硬編碼代理 ---
        elif model_type == "HardcodedBallFollower":
            # obs 的維度定義：(ball_x, ball_y, ball_vx, ball_vy, my_paddle_x, other_paddle_x, spin)
            # 對於 Player A (上板): my_paddle_x 是 top_paddle_x (obs[4])
            # 對於 Player B (下板): my_paddle_x 是 bottom_paddle_x (obs[4])
            # 這由 env._get_obs_for_A() 和 env._get_obs_for_B() 決定，兩者都將自己的 paddle x 放在第4個索引 (0-indexed)

            ball_x = obs[0]
            my_paddle_x = obs[4]
            
            # 為了避免在球拍正下方時抖動，可以加入一個小的容忍區間
            # paddle_width 在環境中定義為 0.2 (正規化座標)
            # 球拍中心點如果已經很接近球的中心點，則不動
            # 這裡我們使用一個簡單的比較，如果需要更精確的追蹤，可以考慮球拍寬度
            # 例如，讓 paddle 中心追到球的 x 座標
            tolerance = 0.01 # 可調整的容忍值，避免過度抖動

            if ball_x < my_paddle_x - tolerance:
                action = 0  # 向左移動
            elif ball_x > my_paddle_x + tolerance:
                action = 2  # 向右移動
            else:
                action = 1  # 保持不動
            return action, None # 硬編碼代理不需要隱藏狀態
        # --- 新增結束 ---
        else:
            raise ValueError(f"未知的模型類型: {model_type}")

# ────────────────── 4. 主循環賽邏輯 ──────────────────
def run_round_robin_tournament(
    env_params: Dict,
    rnn_arch_params: Dict,
    models_to_compete: List[Dict],
    episodes_per_match: int,
    device: torch.device,
    verbose_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """運行循環賽並返回詳細比賽記錄和總結統計。"""

    # 載入所有參賽模型
    print("\n[階段 1/3] 載入所有參賽模型...")
    participants = {}
    for model_info in tqdm(models_to_compete, desc="載入模型"):
        model_name = model_info["name"]
        try:
            participants[model_name] = {
                "model": load_model_universal(model_info, rnn_arch_params, device),
                "type": model_info["type"],
                "path": model_info["path"] # 保留路徑用於日誌
            }
            print(f"  成功載入: {model_name} ({model_info['type']}) จาก {model_info['path']}")
        except Exception as e:
            print(f"  [錯誤] 載入模型 '{model_name}' 失敗: {e}")
            print(f"  跳過此模型。請檢查路徑和 checkpoint 文件。")
    
    active_participants = {name: data for name, data in participants.items() if "model" in data}
    if len(active_participants) < 2:
        print("\n[錯誤] 至少需要兩個成功載入的模型才能進行比賽。請檢查模型配置。")
        # 返回空的 DataFrame
        return pd.DataFrame(), pd.DataFrame(columns=["name", "win", "lose", "draw", "games_played", "win_rate"])


    print(f"\n[階段 2/3] 成功載入 {len(active_participants)} 個模型，開始進行循環賽...")

    # 創建環境 (只需要一個實例)
    current_env_params = dict(env_params) # 複製一份以防修改
    current_env_params["enable_render"] = False # 比賽時不渲染
    env = PongEnv2P(**current_env_params)

    match_records: list[dict] = []
    participant_names = list(active_participants.keys())

    # 創建所有不重複的比賽配對
    match_pairs = list(itertools.combinations(participant_names, 2))
    
    overall_pbar = tqdm(total=len(match_pairs) * episodes_per_match, desc="總比賽進度", unit="局")

    for i in range(len(participant_names)):
        for j in range(i + 1, len(participant_names)):
            name_A = participant_names[i]
            name_B = participant_names[j]

            model_A_data = active_participants[name_A]
            model_B_data = active_participants[name_B]

            net_A, type_A = model_A_data["model"], model_A_data["type"]
            net_B, type_B = model_B_data["model"], model_B_data["type"]
            
            if verbose_progress:
                print(f"\n--- 開始比賽: {name_A} ({type_A}) vs {name_B} ({type_B}) ---")

            # match_pbar = tqdm(range(episodes_per_match), desc=f"比賽: {name_A[:10]} vs {name_B[:10]}", leave=False)
            
            for ep_num in range(episodes_per_match):
                obs_A_curr, obs_B_curr = env.reset()
                done = False
                ep_score_A, ep_score_B = 0, 0

                hidden_A = net_A.init_hidden(1, device) if type_A == "QNetRNN" else None
                hidden_B = net_B.init_hidden(1, device) if type_B == "QNetRNN" else None

                while not done:
                    act_A, hidden_A = select_action_universal(obs_A_curr, net_A, type_A, hidden_A, device)
                    act_B, hidden_B = select_action_universal(obs_B_curr, net_B, type_B, hidden_B, device)
                    
                    (next_obs_A, next_obs_B), (r_A, r_B), done, _ = env.step(act_A, act_B)
                    
                    # 這裡的獎勵是環境原始獎勵，通常是 +1 (得分), -1 (失分), 0 (未結束)
                    # PongEnv2P 的分數在 env.scoreA 和 env.scoreB 中
                    obs_A_curr, obs_B_curr = next_obs_A, next_obs_B
                
                # 一局結束後，根據 env.scoreA 和 env.scoreB 判斷勝者
                current_winner_name = "draw"
                if env.scoreA > env.scoreB: # A (Top) 贏了
                    current_winner_name = name_A
                elif env.scoreB > env.scoreA: # B (Bottom) 贏了
                    current_winner_name = name_B

                match_records.append({
                    "episode": ep_num + 1,
                    "player_A_name": name_A, # Top paddle in env
                    "player_B_name": name_B, # Bottom paddle in env
                    "player_A_type": type_A,
                    "player_B_type": type_B,
                    "score_A": env.scoreA, # 直接使用環境內的最終分數
                    "score_B": env.scoreB,
                    "winner_name": current_winner_name,
                })
                overall_pbar.update(1)
                if verbose_progress:
                    overall_pbar.set_postfix_str(f"局 {ep_num+1}: A {env.scoreA}-{env.scoreB} B, 勝者: {current_winner_name}")
            # match_pbar.close()

    overall_pbar.close()
    env.close()
    print("\n[階段 2/3] 所有比賽完成。")

    # --- 生成統計數據 ---
    print("\n[階段 3/3] 生成統計報告...")
    match_df = pd.DataFrame(match_records)
    
    summary_stats: Dict[str, Dict[str, int]] = {
        name: {"win": 0, "lose": 0, "draw": 0, "games_played": 0} for name in participant_names
    }

    for _, row in match_df.iterrows():
        pA_name, pB_name = row["player_A_name"], row["player_B_name"]
        winner = row["winner_name"]

        summary_stats[pA_name]["games_played"] += 1
        summary_stats[pB_name]["games_played"] += 1

        if winner == pA_name:
            summary_stats[pA_name]["win"] += 1
            summary_stats[pB_name]["lose"] += 1
        elif winner == pB_name:
            summary_stats[pB_name]["win"] += 1
            summary_stats[pA_name]["lose"] += 1
        else: # Draw
            summary_stats[pA_name]["draw"] += 1
            summary_stats[pB_name]["draw"] += 1
            
    summary_df_list = []
    for name, stats in summary_stats.items():
        win_rate = (stats["win"] / stats["games_played"]) if stats["games_played"] > 0 else 0
        summary_df_list.append({
            "name": name,
            "win": stats["win"],
            "lose": stats["lose"],
            "draw": stats["draw"],
            "games_played": stats["games_played"],
            "win_rate": win_rate,
        })
    
    summary_df = pd.DataFrame(summary_df_list).sort_values("win_rate", ascending=False)
    summary_df.set_index("name", inplace=True) # 將 'name' 設為索引

    return match_df, summary_df

# ────────────────── 5. 視覺化函數 ──────────────────
def plot_win_rates(summary_df: pd.DataFrame, output_dir: Path, title_prefix: str = "") -> Path:
    if summary_df.empty:
        print("[警告] summary_df 為空，跳過繪製勝率圖。")
        return output_dir / "win_rates_skipped.png"

    fig, ax = plt.subplots(figsize=(max(8, len(summary_df.index) * 0.8), 6)) # 動態調整寬度
    summary_df["win_rate"].plot.bar(ax=ax, color=sns.color_palette("viridis", len(summary_df.index)))
    ax.set_ylabel("勝率 (Win Rate)")
    ax.set_xlabel("模型 (Model)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title_prefix}循環賽模型勝率排名")
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(summary_df["win_rate"]):
        ax.text(i, v + 0.02, f"{v:.2%}", color='blue', ha='center', fontweight='bold')
    fig.tight_layout()
    out_path = output_dir / "tournament_win_rates.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[信息] 勝率圖已保存到: {out_path}")
    return out_path

def plot_h2h_heatmap(match_df: pd.DataFrame, participant_names: List[str], output_dir: Path, title_prefix: str = "") -> Path:
    if match_df.empty:
        print("[警告] match_df 為空，跳過繪製 H2H 熱力圖。")
        return output_dir / "h2h_heatmap_skipped.png"

    # 創建一個包含所有參賽者的 H2H 表格，初始化為0
    h2h_wins = pd.DataFrame(0, index=participant_names, columns=participant_names, dtype=int)

    for _, row in match_df.iterrows():
        pA, pB, winner = row["player_A_name"], row["player_B_name"], row["winner_name"]
        if winner != "draw":
            # winner 是實際贏得該局的玩家
            # loser 是相對應的另一個玩家
            loser = pB if winner == pA else pA
            if winner in h2h_wins.index and loser in h2h_wins.columns:
                 h2h_wins.at[winner, loser] += 1
            else:
                print(f"[警告] H2H 熱力圖: 找不到玩家 '{winner}' 或 '{loser}'。")


    fig, ax = plt.subplots(figsize=(max(6, len(participant_names) * 0.7), max(5, len(participant_names) * 0.6)))
    sns.heatmap(h2h_wins, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=True, linewidths=.5)
    ax.set_xlabel("輸家 (Loser)")
    ax.set_ylabel("贏家 (Winner)")
    ax.set_title(f"{title_prefix}模型間 Head-to-Head 勝場數")
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    out_path = output_dir / "tournament_h2h_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[信息] H2H 熱力圖已保存到: {out_path}")
    return out_path

# ────────────────── 6. 主執行函數 ──────────────────
def main() -> None:
    start_time = time.time()
    print("="*60)
    print("          Pong 模型循環賽評估器 v2.0")
    print("="*60)

    # --- 載入配置 ---
    env_cfg_path = USER_CONFIG["env_config_path"]
    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_params = yaml.safe_load(f)["env"]

    rnn_cfg_path = USER_CONFIG["rnn_model_config_path"]
    rnn_arch_params = {}
    if Path(rnn_cfg_path).exists():
        with open(rnn_cfg_path, "r", encoding="utf-8") as f:
            rnn_arch_params = yaml.safe_load(f).get("training", {})
    else:
        if any(m["type"] == "QNetRNN" for m in USER_CONFIG["models_to_compete"]):
            print(f"[警告] RNN 配置文件 '{rnn_cfg_path}' 未找到。QNetRNN 模型將使用預設架構參數。")
    
    models_to_compete = USER_CONFIG["models_to_compete"]
    if not models_to_compete or len(models_to_compete) < 2:
        print("[錯誤] 請在 USER_CONFIG 的 'models_to_compete' 中至少配置兩個模型。")
        return

    episodes_per_match = int(USER_CONFIG["episodes_per_match"])
    output_dir = Path(USER_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 使用設備: {device}")
    print(f"[信息] 每對模型將進行 {episodes_per_match} 局比賽。")
    print(f"[信息] 結果將保存到: {output_dir.resolve()}")

    # --- 運行循環賽 ---
    match_df, summary_df = run_round_robin_tournament(
        env_params=env_params,
        rnn_arch_params=rnn_arch_params,
        models_to_compete=models_to_compete,
        episodes_per_match=episodes_per_match,
        device=device,
        verbose_progress=USER_CONFIG["verbose_match_progress"],
    )

    if match_df.empty and summary_df.empty:
        print("\n[信息] 沒有比賽記錄或摘要生成，程序提前結束。")
        return

    # --- 保存結果 ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    match_csv_path = output_dir / f"match_records_{timestamp}.csv"
    summary_csv_path = output_dir / f"summary_ranking_{timestamp}.csv"
    
    match_df.to_csv(match_csv_path, index=False)
    summary_df.to_csv(summary_csv_path)
    print(f"\n[信息] 詳細比賽記錄已保存到: {match_csv_path}")
    print(f"[信息] 總結排名已保存到: {summary_csv_path}")

    # --- 打印最終排名到控制台 ---
    print("\n" + "="*30 + " 最終排名 " + "="*30)
    # 使用制表符或格式化字符串以獲得更整齊的輸出
    header = f"{'Rank':<5} {'Model Name':<25} {'Wins':>5} {'Losses':>6} {'Draws':>5} {'Played':>6} {'Win Rate':>10}"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(summary_df.itertuples(name=None), 1):
        # itertuples() 返回的元組格式為: (index, col1, col2, ...)
        # 第一個元素是索引（模型名稱），後面是各列的值
        model_name = row[0]  # 索引（模型名稱）
        wins = row[1]        # win 列
        losses = row[2]      # lose 列
        draws = row[3]       # draw 列
        played = row[4]      # games_played 列
        win_rate_val = row[5]  # win_rate 列
        print(
            f"#{rank:<4} {model_name:<25} {wins:>5} {losses:>6} {draws:>5} {played:>6} {win_rate_val*100:>9.2f}%"
        )
    print("="* (len(header) + 2))


    # --- 生成圖表 ---
    if USER_CONFIG["generate_plots"]:
        print("\n[信息] 生成圖表中...")
        plot_title_prefix = f"Tournament ({timestamp}) - "
        plot_win_rates(summary_df, output_dir, title_prefix=plot_title_prefix)
        
        # 確保 participant_names 的順序與 summary_df.index 一致，以獲得正確的熱力圖標籤順序
        # 或者直接使用 summary_df.index.tolist()
        plot_h2h_heatmap(match_df, summary_df.index.tolist(), output_dir, title_prefix=plot_title_prefix)
        print("[信息] 圖表生成完畢。")

    total_time = time.time() - start_time
    print(f"\n[信息] 循環賽總耗時: {total_time:.2f} 秒。")
    print("[信息] 程序執行完畢。")

if __name__ == "__main__":
    main()