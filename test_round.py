#!/usr/bin/env python3
"""
round_robin_stats.py ─ Round‑robin tournament evaluator with statistics & plotting
====================================================================
> **更新**：改以手动映射旧架构 checkpoint 到 NoisyNet Dueling QNet，并支持 strict=False 加载。
>   - 在 `load_model` 中对旧的 `fc.*` 键做映射到 `features.*` 及 `fc_A`/`fc_V` μ 参数。
>   - 新架构 checkpoint 仍可直接 strict=True 加载。
>   - 其它功能（统计/绘图）不变。

原始来源：`test_round.py` (用户上传) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
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
from models.qnet import QNet                 # type: ignore

# ╔═════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIG                             ║
# ╚═════════════════════════════════════════════════════════════════╝
        # """"checkpoints/model1-0.pth",
        # "checkpoints/model1-1.pth",
        # "checkpoints/model1-3.pth",
        # "checkpoints/model2-0.pth",
        # "checkpoints/model2-3.pth",
        # "checkpoints/model3-3.pth",
        # "checkpoints/model4-0.pth",
        # "checkpoints/model4-2.pth",
        # "checkpoints/model4-4.pth",
        # "checkpoints/model4-6.pth",
        # "checkpoints/model4-7.pth",
        # "checkpoints/model4-9.pth",
        # "checkpoints/model4-11.pth",
        # "checkpoints/model4-12.pth","""
USER_CONFIG = {
    "config_yaml": "config.yaml",
    "model_paths": [
        "checkpoints/model2-0.pth",
        "checkpoints/model4-12.pth",
    ],
    "episodes_each": 300,
    "output_dir": "results",
    "generate_plots": True,
    "quiet": False,
}

# ----------------------------- 工具函式 -----------------------------

def load_model(model_path: str | Path, device: torch.device) -> QNet:
    """
    Load a QNet checkpoint and return the model in eval mode.
    Supports both new NoisyNet‑Dueling checkpoints and old single‑head fc.* checkpoints.
    
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt.get("modelB"))
    if state is None:
        raise KeyError(f"Checkpoint {path} missing 'model' or 'modelB' key")

    net = QNet(input_dim=7, output_dim=3).to(device)

    # 如果是新架构（NoisyNet+Dueling），直接严格加载
    if any(k.startswith("features.") or k.startswith("fc_V.") or k.startswith("fc_A.") for k in state.keys()):
        net.load_state_dict(state)
    else:
        # 旧架构：键为 fc.0, fc.2, fc.4
        mapped: Dict[str, torch.Tensor] = {}
        # 映射特征层
        for k, v in state.items():
            if k.startswith("fc.0."):
                mapped[k.replace("fc.0.", "features.0.")] = v
            elif k.startswith("fc.2."):
                mapped[k.replace("fc.2.", "features.2.")] = v
        # 把旧最后一层 fc.4 映射到 fc_A μ 参数
        w4 = state["fc.4.weight"]
        b4 = state["fc.4.bias"]
        mapped["fc_A.weight_mu"] = w4
        mapped["fc_A.bias_mu"]   = b4
        # 让 fc_V μ 参数为旧 fc.4 权重/偏置的均值
        mapped["fc_V.weight_mu"] = w4.mean(dim=0, keepdim=True)
        mapped["fc_V.bias_mu"]   = b4.mean().unsqueeze(0)
        # 使用 strict=False 加载，保留 NoisyNet σ & ε 初始值
        net.load_state_dict(mapped, strict=False)

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
    participants: Dict[str, Dict] = {
        Path(p).stem: {"model": load_model(p, device)}
        for p in model_paths
    }

    env_cfg["enable_render"] = False
    env = PongEnv2P(**env_cfg)

    match_records: list[dict] = []
    for a, b in itertools.combinations(participants.keys(), 2):
        net_a = participants[a]["model"]
        net_b = participants[b]["model"]
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
            if score_a > score_b:
                wins_a += 1; winner = a
            elif score_b > score_a:
                wins_b += 1; winner = b
            else:
                winner = "draw"
            match_records.append({
                "episode": ep,
                "player_a": a,
                "player_b": b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
            })
        if verbose:
            print(f"{a} vs {b} → A_wins={wins_a:3}  B_wins={wins_b:3}")

    env.close()

    match_df = pd.DataFrame(match_records)
    wc = match_df[match_df["winner"] != "draw"].groupby("winner").size()
    lc = (
        match_df[match_df["winner"] != "draw"]
        .assign(loser=lambda d: np.where(d["winner"] == d["player_a"], d["player_b"], d["player_a"]))
        .groupby("loser").size()
    )
    summary_df = pd.DataFrame({"win": wc, "lose": lc}).fillna(0).astype(int)
    summary_df["games"] = summary_df["win"] + summary_df["lose"]
    summary_df["win_rate"] = summary_df["win"] / summary_df["games"]
    summary_df = summary_df.sort_values("win_rate", ascending=False)

    return match_df, summary_df

# ---------------------------- 视 觉 化 ----------------------------

def plot_win_rates(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8,5))
    summary_df["win_rate"].plot.bar(ax=ax)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0,1)
    ax.set_title("Tournament Win Rates")
    ax.set_xticklabels(summary_df.index, rotation=45, ha="right")
    fig.tight_layout()
    out_path = out_dir / "win_rates.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_h2h_heatmap(match_df: pd.DataFrame, out_dir: Path) -> Path:
    players = sorted(set(match_df["player_a"]) | set(match_df["player_b"]))
    h2h = pd.DataFrame(0, index=players, columns=players, dtype=int)
    for _, row in match_df.iterrows():
        if row["winner"] == "draw":
            continue
        win, lose = row["winner"], (row["player_b"] if row["winner"] == row["player_a"] else row["player_a"])
        h2h.at[win, lose] += 1
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(h2h, annot=True, fmt="d", cmap="viridis", ax=ax)
    ax.set_xlabel("Loser →"); ax.set_ylabel("Winner →")
    ax.set_title("Head‑to‑Head Wins")
    fig.tight_layout()
    out_path = out_dir / "h2h_heatmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

# ------------------------------- Main -------------------------------

def main() -> None:
    cfg_path = Path(USER_CONFIG["config_yaml"])
    with open(cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)["env"]

    model_paths = USER_CONFIG["model_paths"]
    if not model_paths:
        raise ValueError("请在 USER_CONFIG 中手动列出模型路径！")
    model_paths = [str(p) for p in model_paths]

    episodes_each = int(USER_CONFIG["episodes_each"])
    output_dir = Path(USER_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_plots = bool(USER_CONFIG["generate_plots"])
    quiet = bool(USER_CONFIG["quiet"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match_df, summary_df = round_robin(
        env_cfg=env_cfg,
        model_paths=model_paths,
        episodes_each=episodes_each,
        device=device,
        verbose=not quiet,
    )

    match_csv   = output_dir / "match_records.csv"
    summary_csv = output_dir / "summary.csv"
    match_df.to_csv(match_csv, index=False)
    summary_df.to_csv(summary_csv)
    print(f"\n[✓] 统计完成 → {summary_csv.relative_to(Path.cwd())}")

    if generate_plots:
        win_png = plot_win_rates(summary_df, output_dir)
        h2h_png = plot_h2h_heatmap(match_df, output_dir)
        print("[✓] 图表已生成 →", win_png.relative_to(Path.cwd()), ",", h2h_png.relative_to(Path.cwd()))

    print("\n=== Final Ranking ===")
    for rank, (name, row) in enumerate(summary_df.itertuples(), 1):
        print(
            f"#{rank:2d}  {name:<25}  "
            f"W:{row.win:3d}  L:{row.lose:3d}  WR:{row.win_rate*100:6.2f}%"
        )

if __name__ == "__main__":
    main()
