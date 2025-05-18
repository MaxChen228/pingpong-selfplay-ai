#!/usr/bin/env python3
"""
test.py ─ 2P Pong Viewer
────────────────────────────────────────────────────────────────
• 支援舊 fc.* checkpoint 以及新版 Noisy‑Dueling checkpoint
• 功能：滑桿速度調節、SPACE 暫停、分數 & 球速/轉速顯示、球旋轉圖、軌跡拖曳
"""

from __future__ import annotations

import os, random, time, math
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml
import pygame
import numpy as np
import torch

from envs.my_pong_env_2p import PongEnv2P      # type: ignore
from models.qnet import QNet                   # 新版 (Noisy + Dueling)

# ────────────────── 1. 讀取環境設定 ──────────────────
with open("config.yaml", "r") as f:
    ENV_CFG: Dict[str, Any] = yaml.safe_load(f)["env"]

MODEL_A_PATH = "checkpoints/model2-0.pth"
MODEL_B_PATH = "checkpoints/model4-12.pth"
TEST_EPISODES = 5
BALL_IMG      = "assets/sunglasses.png"      # 你的球圖片

# ────────────────── 2. 載入模型（自動容錯） ──────────────────
def load_model(model_path: str | Path, device: torch.device) -> QNet:
    """
    支援：
      ① 新 checkpoint → features./fc_V./fc_A.  → strict=True
      ② 舊 checkpoint → fc.0./fc.2./fc.4.     → 做 key 映射後 strict=False 讀入
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model") or ckpt.get("modelB")
    if state is None:
        raise KeyError(f"{path} 缺少 'model' 或 'modelB' 欄位 (keys={list(ckpt.keys())})")

    net = QNet(input_dim=7, output_dim=3).to(device)

    # 判斷舊 / 新
    is_new = any(k.startswith(("features.", "fc_V.", "fc_A.")) for k in state)
    if is_new:
        net.load_state_dict(state, strict=True)
    else:
        mapped: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("fc.0."):
                mapped[k.replace("fc.0.", "features.0.")] = v
            elif k.startswith("fc.2."):
                mapped[k.replace("fc.2.", "features.2.")] = v
        w4, b4 = state["fc.4.weight"], state["fc.4.bias"]
        mapped["fc_A.weight_mu"] = w4
        mapped["fc_A.bias_mu"]   = b4
        mapped["fc_V.weight_mu"] = w4.mean(dim=0, keepdim=True)
        mapped["fc_V.bias_mu"]   = b4.mean().unsqueeze(0)
        net.load_state_dict(mapped, strict=False)   # 允許缺少 σ/ε
    net.eval()
    return net

# ────────────────── 3. 選擇動作 (argmax 無探索) ──────────────────
def act_greedy(obs: np.ndarray, net: QNet, dev: torch.device) -> int:
    with torch.no_grad():
        q = net(torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0))
        return int(q.argmax(1).item())

# ────────────────── 4. 主函式 ──────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] Using environment config:", ENV_CFG)
    print("[Info] Model A:", MODEL_A_PATH)
    print("[Info] Model B:", MODEL_B_PATH)

    modelA = load_model(MODEL_A_PATH, device)
    modelB = load_model(MODEL_B_PATH, device)

    env_cfg = dict(ENV_CFG); env_cfg["enable_render"] = False
    env = PongEnv2P(**env_cfg)

    # ── Pygame 初始化 ───────────────────────────
    pygame.init()
    RENDER = env_cfg["render_size"]
    screen = pygame.display.set_mode((RENDER, RENDER))
    pygame.display.set_caption("Pong 2P Viewer")
    font  = pygame.font.SysFont(None, 24)

    # 球圖
    ball_img0 = pygame.image.load(BALL_IMG).convert_alpha()
    ball_img0 = pygame.transform.scale(ball_img0, (20, 20))

    # Slider 參數
    slider_x, slider_y, slider_w, slider_h = 50, 30, 200, 10
    speed_min, speed_max, speed_factor = 0.1, 3.0, 1.0
    knob_x = slider_w * (speed_factor - speed_min) / (speed_max - speed_min)
    paused = False

    def draw_ui(ball_speed: float):
        # slider
        pygame.draw.rect(screen, (180,180,180), (slider_x, slider_y, slider_w, slider_h))
        pygame.draw.rect(screen, (255,100,100), (slider_x+knob_x-5, slider_y-5, 10, 20))
        screen.blit(font.render(f"Speed x{speed_factor:.2f}", True, (255,255,255)),
                    (slider_x+slider_w+15, slider_y-5))
        # 分數
        s_txt = f"A:{env.scoreA}  B:{env.scoreB}"
        screen.blit(font.render(s_txt, True, (255,255,0)), (10,70))
        # 球速 / 轉速
        spin = env.spin
        screen.blit(font.render(f"spin={spin:+5.2f}  v={ball_speed:5.3f}", True, (255,255,255)),
                    (10,100))
        if paused:
            screen.blit(font.render("PAUSED (SPACE)", True, (255,0,0)), (slider_x, slider_y+40))

    # ── 主迴圈 ────────────────────────────────
    for ep in range(TEST_EPISODES):
        obsA, obsB = env.reset()
        spin_angle = 0.0
        trail: List[Tuple[float,float]] = []
        done = False

        while not done:
            # 處理事件
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); return
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                    if pygame.mouse.get_pressed()[0]:
                        mx,my = pygame.mouse.get_pos()
                        if slider_x <= mx <= slider_x+slider_w and slider_y-10 <= my <= slider_y+slider_h+20:
                            knob_x = mx - slider_x
                            speed_factor = speed_min + (speed_max-speed_min)*(knob_x/slider_w)

            if paused:
                screen.fill((30,30,30)); draw_ui(0.0); pygame.display.flip(); time.sleep(0.05)
                continue

            # step
            actA = act_greedy(obsA, modelA, device)
            actB = act_greedy(obsB, modelB, device)
            (obsA, obsB), (_, rB), done, _ = env.step(actA, actB)

            # 旋轉角度 & 軌跡
            spin_angle += env.spin
            bx = env.ball_x * RENDER
            by = env.ball_y * RENDER
            trail.append((bx,by))
            if len(trail) > 30: trail.pop(0)

            # 繪圖
            screen.fill((0,0,0))
            # 擋板
            pw = int(env.paddle_width * RENDER)
            top_x = int(env.top_paddle_x * RENDER)
            bot_x = int(env.bottom_paddle_x * RENDER)
            pygame.draw.rect(screen, (0,255,0), (top_x-pw//2, 0, pw, 10))
            pygame.draw.rect(screen, (0,255,0), (bot_x-pw//2, RENDER-10, pw, 10))
            # 軌跡
            for i in range(1,len(trail)):
                pygame.draw.line(screen, (200,200,200), trail[i-1], trail[i], 2)
            # 球
            rot_ball = pygame.transform.rotate(ball_img0, spin_angle)
            screen.blit(rot_ball, rot_ball.get_rect(center=(bx,by)))
            # UI
            v = math.hypot(env.ball_vx, env.ball_vy)
            draw_ui(v)

            pygame.display.flip()
            time.sleep(0.05 / speed_factor)

        print(f"[Episode {ep+1}] done → B reward = {rB:+.1f}")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
