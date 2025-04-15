#!/usr/bin/env python3

import os
import time
import yaml
import torch
import pygame
import numpy as np

from envs.my_pong_env_2p import PongEnv2P
from models.qnet import QNet

# ---------- 輔助函式：讀取 QNet 權重 ----------
def load_model(model_path, device):
    """
    載入 .pth 檔案，回傳一個 QNet 實例
    預期檔案內含 {'model': state_dict, ...}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_ = QNet(input_dim=7, output_dim=3).to(device)
    if 'model' in checkpoint:
        model_.load_state_dict(checkpoint['model'])
    else:
        # 如果你的檔案內是 'modelA'/'modelB' 之類，請自行調整
        raise KeyError("Checkpoint missing 'model' key. Found keys: " + str(checkpoint.keys()))
    model_.eval()
    return model_

# ---------- 評估階段: 不帶探索，直接 argmax Q ----------
def select_action_eval(obs, model_, device):
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model_(obs_t)
        action = q_values.argmax(dim=1).item()
    return action

def main():
    # ========== 1. 從 config.yaml 讀取參數 ============
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['env']  # 環境相關參數

    # 你可改成 argparse / 參數
    modelA_path = "checkpoints/model_gen0.pth"
    modelB_path = "checkpoints/model_gen0.pth"
    test_episodes = 5  # 觀察對戰回合數

    print("[Info] Using environment config:", env_cfg)
    print(f"[Info] Model A: {modelA_path}")
    print(f"[Info] Model B: {modelB_path}")

    # ========== 2. 建立裝置 & 載入模型 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelA = load_model(modelA_path, device)
    modelB = load_model(modelB_path, device)

    # ========== 3. 建立 Pong 雙人環境 (含剛體+馬格努斯+初始條件range) ============
    env = PongEnv2P(
        render_size  = env_cfg["render_size"],
        paddle_width = env_cfg["paddle_width"],
        paddle_speed = env_cfg["paddle_speed"],
        max_score    = env_cfg["max_score"],
        enable_render= True,

        enable_spin       = env_cfg["enable_spin"],
        magnus_factor     = env_cfg["magnus_factor"],
        restitution       = env_cfg["restitution"],
        friction          = env_cfg["friction"],
        ball_mass         = env_cfg["ball_mass"],
        world_ball_radius = env_cfg["world_ball_radius"],
        ball_angle_intervals = env_cfg["ball_angle_intervals"],
        speed_scale_every = env_cfg["speed_scale_every"],
        speed_increment   = env_cfg["speed_increment"],
        # 讀取 range
        ball_speed_range  = tuple(env_cfg["ball_speed_range"]),
        spin_range        = tuple(env_cfg["spin_range"]) 
    )

    # ========== 4. Pygame GUI 參數 (Slider) ============
    slider_x = 50
    slider_y = 30
    slider_w = 200
    slider_h = 10

    speed_min = 0.1
    speed_max = 3.0
    speed_factor = 1.0  # 初始播放倍率

    def factor_to_knob(factor):
        alpha = (factor - speed_min) / (speed_max - speed_min)
        return slider_w * alpha

    def knob_to_factor(knob_pos):
        alpha = knob_pos / slider_w
        return speed_min + alpha*(speed_max - speed_min)

    knob_x = factor_to_knob(speed_factor)
    paused = False

    # ========== episodes 參數 ===========
    scoreA_total = 0
    scoreB_total = 0

    # ========== 幫助函式：畫 Slider & 文字 ===========
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)

    def draw_slider(surface):
        # bar
        bar_rect = pygame.Rect(slider_x, slider_y, slider_w, slider_h)
        pygame.draw.rect(surface, (180,180,180), bar_rect)

        # knob
        knob_rect = pygame.Rect(slider_x + knob_x -5, slider_y -5, 10, 20)
        pygame.draw.rect(surface, (255,100,100), knob_rect)

        # 文字
        text = font.render(f"Speed x{speed_factor:.2f}", True, (255,255,255))
        surface.blit(text, (slider_x + slider_w + 20, slider_y -5))

        # 顯示暫停
        if paused:
            pause_text = font.render("PAUSED (press SPACE to resume)", True, (255, 0, 0))
            surface.blit(pause_text, (slider_x, slider_y + 30))

    # ========== 5. 開始迴圈，執行 episodes ============
    for ep in range(test_episodes):
        obsA, obsB = env.reset()
        done = False
        episode_scoreA = 0.0
        episode_scoreB = 0.0

        while not done:
            # 處理 Pygame event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[Info] Quit event received.")
                    env.close()
                    pygame.quit()
                    return

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if (slider_x <= mx <= slider_x+slider_w) and (slider_y-10 <= my <= slider_y+slider_h+20):
                        knob_x = mx - slider_x
                        knob_x = max(0, min(slider_w, knob_x))
                        speed_factor = knob_to_factor(knob_x)

                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:
                        mx, my = pygame.mouse.get_pos()
                        if (slider_x <= mx <= slider_x+slider_w) and (slider_y-10 <= my <= slider_y+slider_h+20):
                            knob_x = mx - slider_x
                            knob_x = max(0, min(slider_w, knob_x))
                            speed_factor = knob_to_factor(knob_x)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"[Info] paused={paused}")

            if paused:
                env.render()
                draw_slider(env.screen)
                pygame.display.flip()
                time.sleep(0.05)
                continue

            # A / B 都用 Argmax Q
            actA = select_action_eval(obsA, modelA, device)
            actB = select_action_eval(obsB, modelB, device)

            (nextA, nextB), (rA, rB), done, _ = env.step(actA, actB)
            obsA, obsB = nextA, nextB

            episode_scoreA += rA
            episode_scoreB += rB

            # render & slider
            env.render()
            draw_slider(env.screen)
            pygame.display.flip()

            time.sleep(0.05 / speed_factor)

        print(f"[Episode {ep+1}] A_score={episode_scoreA}, B_score={episode_scoreB}")
        scoreA_total += episode_scoreA
        scoreB_total += episode_scoreB

    env.close()
    pygame.quit()

    print("\n===== Test Finished =====")
    print(f"Total episodes: {test_episodes}")
    print(f"A total score: {scoreA_total}")
    print(f"B total score: {scoreB_total}")
    if scoreB_total > scoreA_total:
        print(">> B wins overall!")
    elif scoreB_total < scoreA_total:
        print(">> A wins overall!")
    else:
        print(">> Tie??")

if __name__ == "__main__":
    main()
