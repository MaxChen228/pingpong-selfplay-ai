#!/usr/bin/env python3
"""
round_robin_stats.py ─ Round‑robin tournament evaluator with statistics & plotting
====================================================================
> **更新**：改以 **`model_paths` 手動清單**，不再自動偵測萬用字元。
>   ‑ 使用者可在 `USER_CONFIG` 填入任意檔案路徑 list。  
>   ‑ 其餘功能 (統計 / 圖表) 不變。

功能一覽
---------
1. **統計所有對戰資料、勝率**  → `match_records.csv`, `summary.csv`
2. **漂亮圖表**                 → `win_rates.png`, `h2h_heatmap.png`
3. **易調整介面 & 中文註解**      → 通通集中在 `USER_CONFIG`  區塊

原始來源：`test_round.py`  (使用者上傳)  citeturn0file0
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from envs.my_pong_env_2p import PongEnv2P  # type: ignore
from models.qnet import QNet  # type: ignore

# ╔═════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIG                             ║
# ║   👇👇👇 只改這裡就行！                                          ║
# ╚═════════════════════════════════════════════════════════════════╝
USER_CONFIG = {
    # YAML 檔：環境設定
    "config_yaml": "config.yaml",

    # ➜ 手動列出所有模型
    "model_paths": [
        "checkpoints/model_gen1.pth",
        "checkpoints/model_gen3.pth",
        "checkpoints/model_gen5.pth",
        "checkpoints/model2_gen0.pth",
        "checkpoints/model2_gen1.pth",
        "checkpoints/model2_gen3.pth",
        "checkpoints/model2_gen5.pth",
        "checkpoints/model3_gen0.pth",
        "checkpoints/model3_gen3.pth",
        "checkpoints/model3_gen4.pth",
        "checkpoints/model4_gen0.pth",
    ],

    # 每兩人對戰的局數
    "episodes_each": 100,

    # 結果輸出資料夾
    "output_dir": "results",

    # 是否產生圖表 (PNG)
    "generate_plots": True,

    # 是否靜音（不顯示每組對戰 log）
    "quiet": False,
}
# ═══════════════════════════════════════════════════════════════════

# ----------------------------- 工具函式 -----------------------------

def load_model(model_path: str | Path, device: torch.device) -> QNet:
    """Load a QNet checkpoint and return the model in eval mode."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    net = QNet(input_dim=7, output_dim=3).to(device)
    try:
        net.load_state_dict(ckpt["model"])
    except KeyError as e:
        raise KeyError(f"Checkpoint {path} missing 'model' key → {e}")
    net.eval()
    return net

def select_action_eval(obs: np.ndarray, model: QNet, device: torch.device) -> int:
    """Greedy action selection (no exploration)."""
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(obs_t)
        action = int(q.argmax(dim=1).item())
    return action

# ---------------------------- 主 tournament ----------------------------

def round_robin(
    env_cfg: Dict,
    model_paths: List[str],
    episodes_each: int,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run round‑robin tournament and return (match_df, summary_df)."""
    # 1. 準備選手 (name → model)
    participants: Dict[str, Dict] = {
        Path(p).stem: {"model": load_model(p, device)} for p in model_paths
    }

    # 2. 建立環境 (統一關閉 render)
    # 強制關閉 render，避免 YAML 內已經帶有同名參數造成重複
    env_cfg["enable_render"] = False
    env = PongEnv2P(**env_cfg)

    match_records: List[Dict] = []

    # 3. 每兩人互打
    for a, b in itertools.combinations(participants.keys(), 2):
        net_a, net_b = participants[a]["model"], participants[b]["model"]
        wins_a = wins_b = 0
        for ep in range(episodes_each):
            obs_a, obs_b = env.reset()
            done = False
            score_a = score_b = 0
            while not done:
                act_a = select_action_eval(obs_a, net_a, device)
                act_b = select_action_eval(obs_b, net_b, device)
                (obs_a, obs_b), (r_a, r_b), done, _ = env.step(act_a, act_b)
                score_a += r_a
                score_b += r_b
            # 記錄結果
            if score_a > score_b:
                wins_a += 1
                winner = a
            elif score_b > score_a:
                wins_b += 1
                winner = b
            else:
                winner = "draw"
            match_records.append(
                {
                    "episode": ep,
                    "player_a": a,
                    "player_b": b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "winner": winner,
                }
            )
        if verbose:
            print(f"{a} vs {b} → A_wins={wins_a:3}  B_wins={wins_b:3}")

    env.close()

    # 4. 統計
    match_df = pd.DataFrame(match_records)
    win_count = match_df[match_df["winner"] != "draw"].groupby("winner").size()
    lose_count = (
        match_df[match_df["winner"] != "draw"]
        .assign(loser=lambda d: np.where(d["winner"] == d["player_a"], d["player_b"], d["player_a"]))
        .groupby("loser")
        .size()
    )
    summary_df = pd.DataFrame({"win": win_count, "lose": lose_count}).fillna(0).astype(int)
    summary_df["games"] = summary_df["win"] + summary_df["lose"]
    summary_df["win_rate"] = summary_df["win"] / summary_df["games"]
    summary_df = summary_df.sort_values("win_rate", ascending=False)

    return match_df, summary_df

# ---------------------------- 視覺化 ----------------------------

def plot_win_rates(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    summary_df["win_rate"].plot.bar(ax=ax)
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 1)
    ax.set_title("Tournament Win Rates")
    ax.set_xticklabels(summary_df.index, rotation=45, ha="right")
    fig.tight_layout()
    out_path = out_dir / "win_rates.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_h2h_heatmap(match_df: pd.DataFrame, out_dir: Path) -> Path:
    players = sorted(set(match_df["player_a"]).union(match_df["player_b"]))
    h2h = pd.DataFrame(0, index=players, columns=players, dtype=int)
    for _, row in match_df.iterrows():
        if row["winner"] == "draw":
            continue
        winner = row["winner"]
        loser = row["player_b"] if winner == row["player_a"] else row["player_a"]
        h2h.loc[winner, loser] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(h2h, annot=True, fmt="d", cmap="viridis", ax=ax)
    ax.set_xlabel("Loser →")
    ax.set_ylabel("Winner →")
    ax.set_title("Head‑to‑Head Wins")
    fig.tight_layout()
    out_path = out_dir / "h2h_heatmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

# ------------------------------- Main -------------------------------

def main() -> None:
    # 讀取環境設定
    cfg_path = Path(USER_CONFIG["config_yaml"])
    with open(cfg_path, "r", encoding="utf‑8") as f:
        env_cfg = yaml.safe_load(f)["env"]

    # 解析 model list (手動指定)
    model_paths = USER_CONFIG["model_paths"]
    if not model_paths:
        raise ValueError("USER_CONFIG['model_paths'] is empty. 請手動列出模型路徑！")
    model_paths = [str(p) for p in model_paths]

    # 其他參數
    episodes_each = int(USER_CONFIG["episodes_each"])
    output_dir = Path(USER_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_plots = bool(USER_CONFIG["generate_plots"])
    quiet = bool(USER_CONFIG["quiet"])

    # 裝置選擇
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 執行比賽
    match_df, summary_df = round_robin(
        env_cfg=env_cfg,
        model_paths=model_paths,
        episodes_each=episodes_each,
        device=device,
        verbose=not quiet,
    )

    # 儲存 CSV 結果
    match_csv = output_dir / "match_records.csv"
    summary_csv = output_dir / "summary.csv"
    match_df.to_csv(match_csv, index=False)
    summary_df.to_csv(summary_csv)
    print(f"\n[✓] 統計完成 → {summary_csv.resolve().relative_to(Path.cwd())}")

    # 產生圖表
    if generate_plots:
        win_png = plot_win_rates(summary_df, output_dir)
        h2h_png = plot_h2h_heatmap(match_df, output_dir)
        print("[✓] 圖表已生成 →", win_png.resolve().relative_to(Path.cwd()), ",", h2h_png.resolve().relative_to(Path.cwd()))

    # 排名顯示
    print("\n=== Final Ranking ===")
    for rank, (name, row) in enumerate(summary_df.itertuples(), 1):
        print(f"#{rank:>2}  {name:<25}  W:{row.win:>3}  L:{row.lose:>3}  WR:{row.win_rate*100:6.2f}%")


if __name__ == "__main__":
    main()

