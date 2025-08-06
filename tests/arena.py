#!/usr/bin/env python3
"""
arena.py ─ 智慧型模型循環賽競技場
====================================================================
版本: 3.0

功能:
- **持久化數據庫**: 使用 `arena_database.json` 記錄所有模型資訊和歷史對戰結果。
- **避免重複計算**: 啟動時會讀取歷史紀錄，只安排尚未完成或未打滿指定局數的比賽。
- **斷點續賽**: 由於每局比賽後即時儲存，程式中斷後可無縫接軌繼續未完成的比賽。
- **簡易模型管理**: 只需在 `ARENA_CONFIG` 中定義候選模型，程式會自動註冊新模型並安排比賽。
- **通用模型載入**: 兼容 QNet (舊式與新式) 和 QNetRNN 模型，並可輕鬆擴充。
- **詳細報告輸出**:
  - 生成包含完整排名的 `summary_ranking.csv`。
  - 生成 Head-to-Head (H2H) 對戰勝場數的熱力圖，視覺化模型間的相剋關係。
"""
from __future__ import annotations

import itertools
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

# (從專案中匯入)
from envs.my_pong_env_2p import PongEnv2P
from models.qnet import QNet
from models.qnet_rnn import QNetRNN

# ────────────────── 1. 用戶配置區域 (ARENA_CONFIG) ──────────────────
# 說明：未來你只需要修改這個區塊就可以加入新模型進行比賽！
# - id: 模型的唯一識別碼，將用於資料庫和報告中。請保持其唯一性。
# - type: 模型類型，必須是 "QNet", "QNetRNN", 或 "HardcodedBallFollower"。
# - path: 模型權重檔案的路徑。對於硬編碼模型，此項無效，可設為 "N/A"。
# - description: (可選) 對此模型的簡短描述。
ARENA_CONFIG: Dict[str, Any] = {
    "env_config_path": "config.yaml",
    "rnn_model_config_path": "config_rnn.yaml",
    "database_path": "arena_database.json",
    "output_dir": "results_arena",
    "episodes_per_match": 100,  # 每一對模型之間應進行的總比賽局數
    "generate_plots": True,

    "candidate_models": [
        {
            "id": "model2-0",
            "type": "QNet",
            "path": "checkpoints/model2-0.pth",
            "description": "Original QNet model, Gen 0"
        },
        {
            "id": "model4-12",
            "type": "QNet",
            "path": "checkpoints/model4-12.pth",
            "description": "Self-play QNet, Gen 12"
        },
        {
            "id": "RNN_Gen1",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_agent_1.pth",
            "description": "4th generation of RNN agent"
        },
        {
            "id": "RNN_Gen2",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_agent_2.pth",
            "description": "4th generation of RNN agent"
        },
        {
            "id": "RNN_Gen3",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_agent_3.pth",
            "description": "4th generation of RNN agent"
        },
        {
            "id": "RNN_Gen4",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_agent_4.pth",
            "description": "4th generation of RNN agent"
        },
        {
            "id": "RNN_Gen5",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_pong_soul_1.pth",
            "description": "5th generation of RNN agent"
        },
        {
            "id": "RNN_Gen6",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_pong_soul_2.pth",
            "description": "6th generation of RNN agent"
        },
        {
            "id": "RNN_Gen7",
            "type": "QNetRNN",
            "path": "checkpoints_rnn/rnn_pong_soul_3.pth",
            "description": "7th generation of RNN agent"
        },
        {
            "id": "BallFollowerBot",
            "type": "HardcodedBallFollower",
            "path": "N/A",
            "description": "A simple bot that follows the ball's x-position"
        },

    ]
}
# ────────────────── (用戶配置區域結束) ──────────────────


# ────────────────── 2. 資料庫管理 ──────────────────
def load_database(db_path: Path) -> Dict[str, List]:
    """安全地載入或初始化 JSON 資料庫"""
    if db_path.exists() and db_path.stat().st_size > 0:
        with open(db_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 確保基礎結構存在
                if 'models' not in data: data['models'] = []
                if 'match_history' not in data: data['match_history'] = []
                return data
            except json.JSONDecodeError:
                print(f"[警告] 資料庫檔案 {db_path} 格式錯誤，將創建新資料庫。")
    return {"models": [], "match_history": []}

def save_database(db_path: Path, data: Dict) -> None:
    """將資料寫入 JSON 資料庫"""
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def register_models(database: Dict, candidates: List[Dict]) -> bool:
    """將候選模型列表中新的模型註冊到資料庫中"""
    registered_ids = {model['id'] for model in database['models']}
    new_models_found = False
    for candidate in candidates:
        if candidate['id'] not in registered_ids:
            print(f"[DEBUG] 註冊新模型: {candidate['id']}")
            database['models'].append(candidate)
            registered_ids.add(candidate['id'])
            new_models_found = True
    return new_models_found

# ────────────────── 3. 模型載入與動作選擇 (與 test_round.py 相似) ──────────────────
def load_model_universal(model_info: Dict[str, str], rnn_arch_config: Dict, device: torch.device) -> Union[nn.Module, str]:
    model_type = model_info["type"]
    model_id = model_info["id"]

    if model_type == "HardcodedBallFollower":
        return "HardcodedAgent"

    path = Path(model_info["path"])
    if not path.exists():
        raise FileNotFoundError(f"模型檔案 '{model_id}' 未找到於: {path}")

    ckpt = torch.load(path, map_location=device)
    state_dict_keys = ["modelB_state", "modelA_state", "modelB", "modelA", "model", "state_dict"]
    state_dict = next((ckpt[key] for key in state_dict_keys if key in ckpt), None)

    if state_dict is None:
        if all(not isinstance(v, dict) for v in ckpt.values()) and any(k.startswith(("fc", "features")) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            raise KeyError(f"在模型 '{model_id}' 的 checkpoint '{path}' 中找不到有效的模型狀態字典。")

    if model_type == "QNet":
        net = QNet(input_dim=7, output_dim=3).to(device)
        net.load_state_dict(state_dict, strict=False)
    elif model_type == "QNetRNN":
        net = QNetRNN(
            input_dim=7, output_dim=3,
            feature_dim=rnn_arch_config.get('feature_dim', 128),
            lstm_hidden_dim=rnn_arch_config.get('lstm_hidden_dim', 128),
            lstm_layers=rnn_arch_config.get('lstm_layers', 1),
            head_hidden_dim=rnn_arch_config.get('head_hidden_dim', 128)
        ).to(device)
        net.load_state_dict(state_dict)
    else:
        raise ValueError(f"不支援的模型類型 '{model_type}' (模型 ID: {model_id})")

    net.eval()
    return net

def select_action_universal(obs: np.ndarray, model: Union[nn.Module, str], model_type: str,
                            hidden_state: Tuple[torch.Tensor, torch.Tensor] | None,
                            device: torch.device) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor] | None]:
    with torch.no_grad():
        if model_type == "QNetRNN":
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            q_values, next_hidden_state = model(obs_tensor, hidden_state)
            return int(q_values.argmax(1).item()), next_hidden_state
        elif model_type == "QNet":
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(obs_tensor)
            return int(q_values.argmax(1).item()), None
        elif model_type == "HardcodedBallFollower":
            ball_x, my_paddle_x = obs[0], obs[4]
            tolerance = 0.02
            if ball_x < my_paddle_x - tolerance: action = 0  # Left
            elif ball_x > my_paddle_x + tolerance: action = 2  # Right
            else: action = 1  # Stay
            return action, None
        else:
            raise ValueError(f"未知的模型類型: {model_type}")

# ────────────────── 4. 核心對戰邏輯 ──────────────────
def create_match_plan(database: Dict, episodes_per_match: int) -> List[Dict]:
    """根據資料庫歷史紀錄，創建需要進行的比賽計畫"""
    model_ids = [m['id'] for m in database['models']]
    match_counts = Counter()

    for record in database['match_history']:
        # 將玩家對排序，以確保 (p1, p2) 和 (p2, p1) 被視為同一場比賽
        pair = tuple(sorted((record['p1'], record['p2'])))
        match_counts[pair] += 1

    match_plan = []
    for p1_id, p2_id in itertools.combinations(model_ids, 2):
        pair = tuple(sorted((p1_id, p2_id)))
        played_count = match_counts[pair]
        episodes_to_run = episodes_per_match - played_count

        if episodes_to_run > 0:
            match_plan.append({
                "p1_id": p1_id,
                "p2_id": p2_id,
                "episodes_to_run": episodes_to_run
            })
    return match_plan

def run_tournament(
    env: PongEnv2P,
    database: Dict,
    db_path: Path,
    match_plan: List[Dict],
    rnn_arch_params: Dict,
    device: torch.device
) -> None:
    """執行比賽計畫中的所有比賽"""
    if not match_plan:
        print("\n[資訊] 所有比賽均已完成，無需執行新對戰。")
        return

    print(f"\n[階段 2/4] 載入比賽模型...")
    # 一次性載入所有需要的模型
    active_model_ids = set(p['p1_id'] for p in match_plan) | set(p['p2_id'] for p in match_plan)
    models_info = {m['id']: m for m in database['models']}
    loaded_models = {}
    for model_id in tqdm(active_model_ids, desc="載入模型"):
        try:
            loaded_models[model_id] = {
                "model": load_model_universal(models_info[model_id], rnn_arch_params, device),
                "type": models_info[model_id]["type"]
            }
        except Exception as e:
            print(f"  [錯誤] 載入模型 '{model_id}' 失敗: {e}")

    total_episodes_to_run = sum(p['episodes_to_run'] for p in match_plan)
    print(f"\n[階段 3/4] 開始執行 {len(match_plan)} 場比賽，共 {total_episodes_to_run} 局...")

    with tqdm(total=total_episodes_to_run, desc="總比賽進度", unit="局") as pbar:
        for match in match_plan:
            id_A, id_B = match['p1_id'], match['p2_id']
            episodes_needed = match['episodes_to_run']
            
            # 跳過載入失敗的模型
            if id_A not in loaded_models or id_B not in loaded_models:
                print(f"[警告] 跳過比賽 {id_A} vs {id_B}，因為至少一個模型載入失敗。")
                pbar.update(episodes_needed) # 更新進度條以反映跳過的局數
                continue

            model_A_data = loaded_models[id_A]
            model_B_data = loaded_models[id_B]
            net_A, type_A = model_A_data["model"], model_A_data["type"]
            net_B, type_B = model_B_data["model"], model_B_data["type"]
            
            pbar.set_description(f"比賽: {id_A[:10]} vs {id_B[:10]}")

            for _ in range(episodes_needed):
                obs_A, obs_B = env.reset()
                done = False
                
                hidden_A = net_A.init_hidden(1, device) if type_A == "QNetRNN" else None
                hidden_B = net_B.init_hidden(1, device) if type_B == "QNetRNN" else None

                while not done:
                    act_A, hidden_A = select_action_universal(obs_A, net_A, type_A, hidden_A, device)
                    act_B, hidden_B = select_action_universal(obs_B, net_B, type_B, hidden_B, device)
                    (obs_A, obs_B), _, done, _ = env.step(act_A, act_B)

                winner = "draw"
                if env.scoreA > env.scoreB: winner = id_A
                elif env.scoreB > env.scoreA: winner = id_B

                # 即時記錄並儲存
                database["match_history"].append({
                    "p1": id_A,
                    "p2": id_B,
                    "winner": winner,
                    "p1_score": env.scoreA,
                    "p2_score": env.scoreB,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
                save_database(db_path, database)
                pbar.update(1)

# ────────────────── 5. 報告與視覺化 ──────────────────
def generate_summary_report(database: Dict) -> pd.DataFrame:
    """從完整的資料庫歷史紀錄中生成摘要統計"""
    model_ids = [m['id'] for m in database['models']]
    stats = {mid: {"win": 0, "lose": 0, "draw": 0} for mid in model_ids}

    for record in database['match_history']:
        p1, p2, winner = record['p1'], record['p2'], record['winner']
        if winner == "draw":
            stats[p1]["draw"] += 1
            stats[p2]["draw"] += 1
        elif winner == p1:
            stats[p1]["win"] += 1
            stats[p2]["lose"] += 1
        elif winner == p2:
            stats[p2]["win"] += 1
            stats[p1]["lose"] += 1

    summary_list = []
    for mid, s in stats.items():
        games_played = s["win"] + s["lose"] + s["draw"]
        win_rate = s["win"] / games_played if games_played > 0 else 0
        summary_list.append({
            "model_id": mid,
            "win": s["win"],
            "lose": s["lose"],
            "draw": s["draw"],
            "games_played": games_played,
            "win_rate": win_rate,
        })

    summary_df = pd.DataFrame(summary_list).sort_values("win_rate", ascending=False)
    return summary_df.set_index("model_id")

def plot_h2h_heatmap(database: Dict, output_path: Path):
    """生成 Head-to-Head 對戰勝場數的熱力圖"""
    model_ids = [m['id'] for m in database['models']]
    h2h_wins = pd.DataFrame(0, index=model_ids, columns=model_ids)

    for record in database['match_history']:
        winner = record.get('winner')
        if winner != "draw":
            p1, p2 = record['p1'], record['p2']
            loser = p2 if winner == p1 else p1
            h2h_wins.at[winner, loser] += 1

    plt.figure(figsize=(max(8, len(model_ids)), max(6, len(model_ids) * 0.8)))
    sns.heatmap(h2h_wins, annot=True, fmt="d", cmap="viridis", cbar=True, linewidths=.5)
    plt.xlabel("Loser")
    plt.ylabel("Winner")
    plt.title("Head-to-Head (H2H) Match Wins Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[圖表] H2H 熱力圖已儲存至: {output_path}")

# ────────────────── 6. 主執行函數 ──────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("       🏓 Pong AI 智慧型模型競技場 v3.0 🏓")
    print("=" * 60)

    # --- 讀取配置 ---
    output_dir = Path(ARENA_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(ARENA_CONFIG["database_path"])

    with open(ARENA_CONFIG["env_config_path"], "r") as f:
        env_params = yaml.safe_load(f)["env"]
        env_params["enable_render"] = False

    rnn_arch_params = {}
    if Path(ARENA_CONFIG["rnn_model_config_path"]).exists():
        with open(ARENA_CONFIG["rnn_model_config_path"], "r") as f:
            rnn_arch_params = yaml.safe_load(f).get("training", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[資訊] 使用設備: {device}")
    print(f"[資訊] 資料庫路徑: {db_path.resolve()}")
    print(f"[資訊] 目標對戰局數: {ARENA_CONFIG['episodes_per_match']} 局/每對組合")

    # --- [階段 1/4] 載入資料庫與註冊模型 ---
    print("\n[階段 1/4] 載入資料庫並註冊新模型...")
    database = load_database(db_path)
    if register_models(database, ARENA_CONFIG["candidate_models"]):
        print("[資訊] 發現並註冊了新模型，正在更新資料庫...")
        save_database(db_path, database)
    else:
        print("[資訊] 無新模型需要註冊。")

    # --- [階段 2/4 & 3/4] 創建比賽計畫並執行 ---
    match_plan = create_match_plan(database, ARENA_CONFIG['episodes_per_match'])
    if match_plan:
        env = PongEnv2P(**env_params)
        run_tournament(env, database, db_path, match_plan, rnn_arch_params, device)
        env.close()
    else:
        print("\n[階段 2/4 & 3/4] 無需進行新的比賽。")


    # --- [階段 4/4] 生成最終報告 ---
    print("\n[階段 4/4] 生成最終比賽報告...")
    final_summary_df = generate_summary_report(database)

    # 儲存 CSV
    summary_csv_path = output_dir / f"summary_ranking_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    final_summary_df.to_csv(summary_csv_path)
    print(f"\n[報告] 最終排名已儲存至: {summary_csv_path}")

    # 打印最終排名到控制台
    print("\n" + "="*35 + " 最終排名 " + "="*35)
    print(final_summary_df.to_string(columns=["win", "lose", "draw", "games_played", "win_rate"], float_format="{:.2%}".format))
    print("=" * (len(final_summary_df.to_string(header=False).split('\n')[0]) + 15) )


    # 生成圖表
    if ARENA_CONFIG["generate_plots"]:
        h2h_path = output_dir / f"h2h_heatmap_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_h2h_heatmap(database, h2h_path)

    total_time = time.time() - start_time
    print(f"\n[完成] 競技場運行總耗時: {total_time:.2f} 秒。")

if __name__ == "__main__":
    main()