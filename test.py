#!/usr/bin/env python3

import os
import time
import math
import yaml
import pygame
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

    model_ = QNet(input_dim=7, output_dim=3).to(device)
    if 'model' in checkpoint:
        model_.load_state_dict(checkpoint['model'])
    else:
        raise KeyError("Checkpoint missing 'model' key. Found keys: " + str(checkpoint.keys()))
    model_.eval()
    return model_

def select_action_eval(obs, model_, device):
    """
    不帶探索 => 直接 argmax Q
    """
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model_(obs_t)
        action = q_values.argmax(dim=1).item()
    return action

def main():
    # 1) 讀取 config.yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['env']

    modelA_path = "checkpoints/model2_gen5.pth"
    modelB_path = "checkpoints/model3_gen0.pth"
    test_episodes = 5

    print("[Info] Using environment config:", env_cfg)
    print(f"[Info] Model A: {modelA_path}")
    print(f"[Info] Model B: {modelB_path}")

    # 2) 建立裝置 & 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelA = load_model(modelA_path, device)
    modelB = load_model(modelB_path, device)

    # 3) 建立 Pong 雙人環境 => enable_render=False, 我們要自行繪圖
    env = PongEnv2P(
        render_size  = env_cfg["render_size"],
        paddle_width = env_cfg["paddle_width"],
        paddle_speed = env_cfg["paddle_speed"],
        max_score    = env_cfg["max_score"],
        enable_render= False,

        enable_spin       = env_cfg["enable_spin"],
        magnus_factor     = env_cfg["magnus_factor"],
        restitution       = env_cfg["restitution"],
        friction          = env_cfg["friction"],
        ball_mass         = env_cfg["ball_mass"],
        world_ball_radius = env_cfg["world_ball_radius"],
        ball_angle_intervals = env_cfg["ball_angle_intervals"],
        speed_scale_every = env_cfg["speed_scale_every"],
        speed_increment   = env_cfg["speed_increment"],
        ball_speed_range  = tuple(env_cfg["ball_speed_range"]),
        spin_range        = tuple(env_cfg["spin_range"])
    )

    # 4) 準備 Pygame 視窗
    pygame.init()
    screen = pygame.display.set_mode((env_cfg["render_size"], env_cfg["render_size"]))
    pygame.display.set_caption("2P Pong Test + Scoreboard + Sliders + Spin + Trail")
    clock = pygame.time.Clock()

    # 載入球圖片, 並縮放到和原球類似大小
    ball_img_orig = pygame.image.load("assets/sunglasses.png").convert_alpha()
    ball_img_orig = pygame.transform.scale(ball_img_orig, (20, 20))  # 大小可調

    # 4-1) Slider 參數
    slider_x = 50
    slider_y = 30
    slider_w = 200
    slider_h = 10
    speed_min = 0.1
    speed_max = 3.0
    speed_factor = 1.0
    def factor_to_knob(factor):
        alpha = (factor - speed_min) / (speed_max - speed_min)
        return slider_w * alpha
    def knob_to_factor(knob_pos):
        alpha = knob_pos / slider_w
        return speed_min + alpha*(speed_max - speed_min)
    knob_x = factor_to_knob(speed_factor)
    paused = False

    # 4-2) 字體 & scoreboard
    font = pygame.font.SysFont(None, 24)

    # 4-3) 我們自己維護 spinAngle
    spinAngle = 0.0

    # 4-4) 軌跡長度
    TRAIL_SIZE = 30

    def draw_slider_and_info():
        # slider bar
        bar_rect = pygame.Rect(slider_x, slider_y, slider_w, slider_h)
        pygame.draw.rect(screen, (180,180,180), bar_rect)
        # knob
        knob_rect = pygame.Rect(slider_x + knob_x -5, slider_y -5, 10, 20)
        pygame.draw.rect(screen, (255,100,100), knob_rect)
        # 文字
        text = font.render(f"Speed x{speed_factor:.2f}", True, (255,255,255))
        screen.blit(text, (slider_x + slider_w + 20, slider_y -5))

        # scoreboard => 顯示 A/B 分數
        scoreboard_txt = f"A Score: {env.scoreA}   B Score: {env.scoreB}"
        scoreboard_surf = font.render(scoreboard_txt, True, (255,255,0))
        screen.blit(scoreboard_surf, (10, 70))

        # 顯示球速 & 轉速
        ball_speed = math.hypot(env.ball_vx, env.ball_vy)
        spin_text = f"Ball spin={env.spin:.2f}, speed={ball_speed:.3f}"
        spin_surf = font.render(spin_text, True, (255,255,255))
        screen.blit(spin_surf, (10, 100))

        # 顯示暫停
        if paused:
            pause_text = font.render("PAUSED (press SPACE)", True, (255, 0, 0))
            screen.blit(pause_text, (slider_x, slider_y + 40))

    # 5) episodes 參數
    scoreA_total = 0
    scoreB_total = 0

    # 6) 主迴圈
    for ep in range(test_episodes):
        obsA, obsB = env.reset()
        done = False
        episode_scoreA = 0
        episode_scoreB = 0
        spinAngle = 0.0

        # 每回合開始 => 清空軌跡
        ball_trail = []

        while not done:
            # (A) 處理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = pygame.mouse.get_pos()
                    if (slider_x <= mx <= slider_x+slider_w) and (slider_y-10 <= my <= slider_y+slider_h+20):
                        knob_x = mx - slider_x
                        knob_x = max(0, min(slider_w, knob_x))
                        speed_factor = knob_to_factor(knob_x)
                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:
                        mx,my = pygame.mouse.get_pos()
                        if (slider_x <= mx <= slider_x+slider_w) and (slider_y-10 <= my <= slider_y+slider_h+20):
                            knob_x = mx - slider_x
                            knob_x = max(0, min(slider_w, knob_x))
                            speed_factor = knob_to_factor(knob_x)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            # (B) 若暫停 => 不做 step
            if paused:
                screen.fill((30,30,30))
                draw_slider_and_info()
                pygame.display.flip()
                time.sleep(0.05)
                continue

            # (C) 用 modelA, modelB 做動作 => env.step
            actA = select_action_eval(obsA, modelA, device)
            actB = select_action_eval(obsB, modelB, device)
            (nextA,nextB), (rA, rB), done, _ = env.step(actA, actB)
            obsA, obsB = nextA, nextB
            episode_scoreA += rA
            episode_scoreB += rB

            # (D) 根據 spin 累加 angle => spinAngle += env.spin
            spinAngle += env.spin

            # (E) 紀錄球位置到 trail
            bx = env.ball_x * env.render_size
            by = env.ball_y * env.render_size
            ball_trail.append((bx,by))
            # 控制最大長度
            if len(ball_trail) > TRAIL_SIZE:
                ball_trail.pop(0)

            # (F) 繪圖
            screen.fill((0,0,0))

            # 1. 畫擋板
            tx = int(env.top_paddle_x * env.render_size)
            pw = int(env.paddle_width * env.render_size)
            pygame.draw.rect(screen, (0,255,0), (tx - pw//2, 0, pw, 10))

            bx_ = int(env.bottom_paddle_x * env.render_size)
            pygame.draw.rect(screen, (0,255,0), (bx_ - pw//2, env.render_size-10, pw, 10))

            # 2. 畫球軌跡
            #   連結 ball_trail[i], ball_trail[i+1]
            for i in range(1, len(ball_trail)):
                p0 = ball_trail[i-1]
                p1 = ball_trail[i]
                pygame.draw.line(screen, (200,200,200), p0, p1, 2)

            # 3. 畫球 (用圖檔 + rotate)
            #   畫在 ball_trail[-1] (當前位置)
            if len(ball_trail)>0:
                bxPos, byPos = ball_trail[-1]
                rot_ball = pygame.transform.rotate(ball_img_orig, spinAngle)
                rect = rot_ball.get_rect(center=(bxPos, byPos))
                screen.blit(rot_ball, rect)

            # 4. 畫 slider + scoreboard + ball info
            draw_slider_and_info()

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
