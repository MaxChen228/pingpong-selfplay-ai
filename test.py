#!/usr/bin/env python3
"""
test.py ─ 2P Pong Viewer (兼容 QNet 與 QNetRNN 模型)
────────────────────────────────────────────────────────────────
功能：
- 支持載入 QNet (舊式 fc.* 或新式 NoisyDueling) 和 QNetRNN 模型。
- 可在 USER_CONFIG 中方便地指定模型A和模型B的路徑及類型。
- 滑桿速度調節、SPACE 暫停繼續。
- 顯示分數、球的線速度和角速度 (spin)。
- 使用圖片作為球，並根據旋轉角度旋轉球的圖片。
- 顯示球的運動軌跡。
"""

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union # Union for model type

import yaml
import pygame
import numpy as np
import torch
import torch.nn as nn # For type hinting nn.Module

from envs.my_pong_env_2p import PongEnv2P
from models.qnet import QNet
from models.qnet_rnn import QNetRNN

# ────────────────── 1. 用戶配置區域 (USER_CONFIG) ──────────────────
USER_CONFIG: Dict[str, Any] = {
    "env_config_path": "config.yaml",       # 環境配置文件路徑 (主要用於環境參數)
    "rnn_model_config_path": "config_rnn.yaml", # RNN 模型架構參數配置文件路徑 (如果載入RNN模型會用到)

    "model_a": {
        "name": "Model_A_QNet",             # 模型 A 的名稱 (顯示用)
        "path": "checkpoints/model4-0.pth", # 模型 A 的 checkpoint 路徑
        "type": "QNet",                     # 模型類型: "QNet" 或 "QNetRNN"
    },
    "model_b": {
        "name": "Model_B_RNN",              # 模型 B 的名稱 (顯示用)
        "path": "checkpoints_rnn/rnn_agent_4.pth", # 模型 B 的 checkpoint 路徑
        "type": "QNetRNN",                  # 模型類型: "QNet" 或 "QNetRNN"
    },
    "test_episodes": 10,                     # 測試的總局數
    "ball_image_path": "assets/sunglasses.png", # 球的圖片路徑 (例如 sunglasses.png)
                                             # 請確保此圖片存在，或者設為 None 不使用圖片
    "render_fps": 60,                       # 渲染的目標 FPS
    "trail_length": 40,                     # 球軌跡的長度
    "enable_render": True,                  # 是否啟用 Pygame 渲染
}
# ────────────────── (用戶配置區域結束) ──────────────────


# ────────────────── 2. 載入模型（自動兼容 QNet 和 QNetRNN） ──────────────────
def load_model_universal(model_info: Dict[str, str],
                         rnn_arch_config: Dict[str, Any],
                         device: torch.device) -> nn.Module:
    """
    通用模型載入函數，支持 QNet 和 QNetRNN。
    - model_info: 包含 "path" 和 "type" 的字典。
    - rnn_arch_config: 從 rnn_config.yaml 讀取的 'training' 部分，用於 QNetRNN 實例化。
    - device: torch.device。
    """
    model_path_str = model_info["path"]
    model_type = model_info["type"]
    
    path = Path(model_path_str)
    if not path.exists():
        raise FileNotFoundError(f"模型文件未找到: {path}")

    ckpt = torch.load(path, map_location=device)
    
    # 嘗試從常見的鍵名獲取模型狀態字典
    # QNet/QNetRNN 訓練腳本中可能使用 model, modelA, modelB, modelA_state, modelB_state 等
    state_dict_keys_to_try = [
        "modelB_state", "modelA_state", "modelB", "modelA", "model", "state_dict"
    ]
    state_dict = None
    for key in state_dict_keys_to_try:
        if key in ckpt:
            state_dict = ckpt[key]
            break
    
    if state_dict is None:
        # 如果是 QNet 的舊格式（直接是 state_dict）
        if all(not isinstance(v, dict) for v in ckpt.values()) and any (k.startswith("fc.") or k.startswith("features.") for k in ckpt.keys()):
             state_dict = ckpt
        else:
            raise KeyError(f"在 checkpoint '{path}' 中找不到有效的模型狀態字典。已嘗試的鍵名: {state_dict_keys_to_try}，以及直接作為 state_dict。ckpt keys: {list(ckpt.keys())}")


    input_dim = 7  # PongEnv2P 觀察空間維度
    output_dim = 3 # PongEnv2P 動作空間維度

    if model_type == "QNet":
        net = QNet(input_dim=input_dim, output_dim=output_dim).to(device)
        # 處理 QNet 的不同 checkpoint 格式 (舊式 fc vs 新式 NoisyDueling)
        is_new_qnet_format = any(k.startswith(("features.", "fc_V.", "fc_A.")) for k in state_dict)
        if is_new_qnet_format:
            net.load_state_dict(state_dict, strict=True)
        else: # 舊式 QNet (fc.0, fc.2, fc.4)
            mapped_state_dict: Dict[str, torch.Tensor] = {}
            for k, v_tensor in state_dict.items():
                if k.startswith("fc.0."):
                    mapped_state_dict[k.replace("fc.0.", "features.0.")] = v_tensor
                elif k.startswith("fc.2."):
                    mapped_state_dict[k.replace("fc.2.", "features.2.")] = v_tensor
            
            # 處理最後一層 fc.4 到 Dueling Heads 的映射
            if "fc.4.weight" in state_dict and "fc.4.bias" in state_dict:
                w4, b4 = state_dict["fc.4.weight"], state_dict["fc.4.bias"]
                mapped_state_dict["fc_A.weight_mu"] = w4
                mapped_state_dict["fc_A.bias_mu"] = b4
                mapped_state_dict["fc_V.weight_mu"] = w4.mean(dim=0, keepdim=True) # V的權重是A權重的均值
                mapped_state_dict["fc_V.bias_mu"] = b4.mean().unsqueeze(0)      # V的偏置是A偏置的均值
            else:
                print(f"[警告] 舊式 QNet checkpoint '{path}' 缺少 'fc.4.weight' 或 'fc.4.bias'，可能無法完全載入Dueling Head。")

            net.load_state_dict(mapped_state_dict, strict=False) # strict=False 以允許Noisy層的sigma/epsilon缺失
            print(f"[信息] 已將舊式 QNet checkpoint '{path}' 映射到 NoisyDueling 架構 (strict=False)。")

    elif model_type == "QNetRNN":
        net = QNetRNN(
            input_dim=input_dim,
            output_dim=output_dim,
            feature_dim=rnn_arch_config.get('feature_dim', 128),
            lstm_hidden_dim=rnn_arch_config.get('lstm_hidden_dim', 128),
            lstm_layers=rnn_arch_config.get('lstm_layers', 1),
            head_hidden_dim=rnn_arch_config.get('head_hidden_dim', 128)
        ).to(device)
        net.load_state_dict(state_dict)
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

    net.eval() # 設置為評估模式
    if hasattr(net, 'reset_noise'): # 如果模型有 reset_noise (例如 NoisyNet)，在評估前重置一次
        net.reset_noise()
    return net

# ────────────────── 3. 選擇動作函數 ──────────────────
def select_action(obs: np.ndarray, model: nn.Module, model_type: str,
                  hidden_state: Tuple[torch.Tensor, torch.Tensor] | None,
                  device: torch.device) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor] | None]:
    """
    根據模型類型選擇動作。
    對於 QNetRNN，需要傳入和傳出 hidden_state。
    對於 QNet，hidden_state 為 None。
    """
    with torch.no_grad():
        if model_type == "QNetRNN":
            assert hidden_state is not None, "QNetRNN 需要 hidden_state"
            assert isinstance(model, QNetRNN), "模型實例與類型 'QNetRNN' 不符"
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # (batch=1, seq=1, features)
            q_values, next_hidden_state = model(obs_tensor, hidden_state)
            action = int(q_values.argmax(1).item())
            return action, next_hidden_state
        elif model_type == "QNet":
            assert isinstance(model, QNet), "模型實例與類型 'QNet' 不符"
            if hasattr(model, 'reset_noise'): # NoisyNet 在每次前向傳播前重置噪聲 (即使是評估時，如果這是期望的行為)
                model.reset_noise() # 注意：原 test.py 在 act_greedy 中不重置噪聲，這裡保持一致性，評估時不主動重置
                                     # 但如果 QNet 的 forward 預期在 eval 時不使用噪聲，NoisyLinear 層會自動處理。
                                     # 如果你想在每次評估選擇動作時都用新的固定噪聲（雖然不常見），可以取消註釋 reset_noise()。
                pass

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(obs_tensor)
            action = int(q_values.argmax(1).item())
            return action, None # QNet 不需要 hidden_state
        else:
            raise ValueError(f"未知的模型類型: {model_type}")


# ────────────────── 4. Pygame UI 繪製函數 ──────────────────
class GameUI:
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font, render_size: int, ball_image_path: str | None):
        self.screen = screen
        self.font = font
        self.render_size = render_size
        self.ball_original_image = None
        if ball_image_path and Path(ball_image_path).exists():
            try:
                self.ball_original_image = pygame.image.load(ball_image_path).convert_alpha()
                # 調整球圖片大小，例如調整到直徑約為球半徑的兩倍像素
                # PongEnv2P 中的 world_ball_radius 預設是 0.03，乘以 render_size 得到像素半徑
                # 這裡假設一個固定大小，或從 env 實例獲取 ball_radius
                image_diameter = int(0.03 * 2 * self.render_size * 1.5) # 稍微放大一點
                self.ball_original_image = pygame.transform.scale(self.ball_original_image, (image_diameter, image_diameter))
            except pygame.error as e:
                print(f"[警告] 無法載入球圖片 '{ball_image_path}': {e}")
                self.ball_original_image = None
        else:
            if ball_image_path: # 如果提供了路徑但文件不存在
                 print(f"[警告] 球圖片 '{ball_image_path}' 未找到。將使用預設圓形繪製球。")

        # Slider 參數
        self.slider_rect = pygame.Rect(50, self.render_size - 40, 200, 10) # 位置調整到下方
        self.knob_width = 10
        self.knob_height = 20
        self.speed_min, self.speed_max = 0.1, 5.0 # 調整速度範圍
        self.current_speed_factor = 1.0
        self.knob_x_offset = (self.slider_rect.width - self.knob_width) * \
                             (self.current_speed_factor - self.speed_min) / (self.speed_max - self.speed_min)

    def update_slider(self, mouse_pos: Tuple[int, int], mouse_pressed: bool) -> None:
        if mouse_pressed and self.slider_rect.collidepoint(mouse_pos[0], mouse_pos[1] + self.knob_height // 2): # 擴大一點點滑動區域
            self.knob_x_offset = mouse_pos[0] - self.slider_rect.x - self.knob_width / 2
            self.knob_x_offset = max(0, min(self.knob_x_offset, self.slider_rect.width - self.knob_width))
            self.current_speed_factor = self.speed_min + \
                                       (self.speed_max - self.speed_min) * \
                                       (self.knob_x_offset / (self.slider_rect.width - self.knob_width))
            self.current_speed_factor = max(self.speed_min, min(self.current_speed_factor, self.speed_max))


    def draw(self, env: PongEnv2P, model_a_name: str, model_b_name: str, paused: bool, trail: List[Tuple[float, float]], spin_angle: float):
        self.screen.fill((30, 30, 30)) # 背景色

        # 繪製擋板
        paddle_pixel_width = int(env.paddle_width * self.render_size)
        paddle_pixel_height = 10
        # 上方擋板 (A)
        top_paddle_center_x = int(env.top_paddle_x * self.render_size)
        pygame.draw.rect(self.screen, (0, 200, 0),
                         (top_paddle_center_x - paddle_pixel_width // 2, 0, paddle_pixel_width, paddle_pixel_height))
        # 下方擋板 (B)
        bottom_paddle_center_x = int(env.bottom_paddle_x * self.render_size)
        pygame.draw.rect(self.screen, (0, 200, 0),
                         (bottom_paddle_center_x - paddle_pixel_width // 2, self.render_size - paddle_pixel_height,
                          paddle_pixel_width, paddle_pixel_height))

        # 繪製軌跡
        if len(trail) > 1:
            pygame.draw.aalines(self.screen, (150, 150, 150), False, trail)

        # 繪製球
        ball_pixel_x = int(env.ball_x * self.render_size)
        ball_pixel_y = int(env.ball_y * self.render_size)
        if self.ball_original_image:
            rotated_ball_image = pygame.transform.rotate(self.ball_original_image, -spin_angle) # Pygame 角度是逆時針
            ball_rect = rotated_ball_image.get_rect(center=(ball_pixel_x, ball_pixel_y))
            self.screen.blit(rotated_ball_image, ball_rect)
        else: # 預設繪製圓形
            ball_pixel_radius = int(env.world_ball_radius * self.render_size)
            pygame.draw.circle(self.screen, (255, 255, 255), (ball_pixel_x, ball_pixel_y), ball_pixel_radius)


        # 繪製 Slider
        pygame.draw.rect(self.screen, (180, 180, 180), self.slider_rect)
        knob_rect = pygame.Rect(self.slider_rect.x + self.knob_x_offset,
                                self.slider_rect.y - (self.knob_height - self.slider_rect.height) // 2,
                                self.knob_width, self.knob_height)
        pygame.draw.rect(self.screen, (255, 100, 100), knob_rect)
        
        speed_text = self.font.render(f"Speed x{self.current_speed_factor:.2f}", True, (220, 220, 220))
        self.screen.blit(speed_text, (self.slider_rect.right + 10, self.slider_rect.centery - speed_text.get_height() // 2))

        # 繪製分數和狀態信息
        score_text = self.font.render(f"{model_a_name} (Top): {env.scoreA}  -  {model_b_name} (Bot): {env.scoreB}", True, (255, 255, 0))
        self.screen.blit(score_text, (10, 10))

        ball_speed_val = math.hypot(env.ball_vx, env.ball_vy)
        # 將速度和旋轉正規化以便更好地理解（可選）
        # normalized_speed = ball_speed_val / math.hypot(env.ball_speed_range[1], env.ball_speed_range[1]) # 相對於最大初始速度
        # normalized_spin = env.spin / env.spin_range[1] # 相對於最大初始旋轉
        status_text_line1 = self.font.render(f"Ball Speed: {ball_speed_val:.3f}", True, (220, 220, 220))
        status_text_line2 = self.font.render(f"Ball Spin: {env.spin:+.2f} rad/step", True, (220, 220, 220))
        self.screen.blit(status_text_line1, (10, 35))
        self.screen.blit(status_text_line2, (10, 60))


        if paused:
            pause_text = self.font.render("PAUSED (SPACE to continue)", True, (255, 0, 0))
            pause_rect = pause_text.get_rect(center=(self.render_size // 2, self.render_size // 2))
            self.screen.blit(pause_text, pause_rect)

        pygame.display.flip()


# ────────────────── 5. 主函式 ──────────────────
def main() -> None:
    # --- 初始化 ---
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 使用設備: {device}")

    # --- 載入配置 ---
    env_config_path = USER_CONFIG["env_config_path"]
    with open(env_config_path, "r") as f:
        env_params = yaml.safe_load(f)["env"]
        env_params["enable_render"] = False # Pygame 由此腳本控制渲染

    rnn_model_config_path = USER_CONFIG["rnn_model_config_path"]
    rnn_arch_config = {}
    if Path(rnn_model_config_path).exists():
        with open(rnn_model_config_path, "r") as f:
            rnn_arch_config = yaml.safe_load(f)["training"] # QNetRNN 架構參數在 'training' 下
    elif USER_CONFIG["model_a"]["type"] == "QNetRNN" or USER_CONFIG["model_b"]["type"] == "QNetRNN":
        print(f"[警告] RNN 模型配置文件 '{rnn_model_config_path}' 未找到，將使用 QNetRNN 的預設架構參數。")


    # --- 創建環境 ---
    env = PongEnv2P(**env_params)
    render_size = env_params.get("render_size", 400) # 從環境配置獲取渲染大小

    # --- 載入模型 ---
    print(f"[信息] 載入模型 A ({USER_CONFIG['model_a']['type']}): {USER_CONFIG['model_a']['path']}")
    model_a = load_model_universal(USER_CONFIG["model_a"], rnn_arch_config, device)
    
    print(f"[信息] 載入模型 B ({USER_CONFIG['model_b']['type']}): {USER_CONFIG['model_b']['path']}")
    model_b = load_model_universal(USER_CONFIG["model_b"], rnn_arch_config, device)

    # --- Pygame 設置 ---
    screen = None
    game_ui = None
    if USER_CONFIG["enable_render"]:
        screen = pygame.display.set_mode((render_size, render_size))
        pygame.display.set_caption(f"Pong Viewer: {USER_CONFIG['model_a']['name']} vs {USER_CONFIG['model_b']['name']}")
        try:
            font = pygame.font.SysFont("Consolas", 20) #嘗試使用 Consolas
        except:
            font = pygame.font.SysFont(None, 24) # 備用字體
        game_ui = GameUI(screen, font, render_size, USER_CONFIG["ball_image_path"])

    clock = pygame.time.Clock()
    paused = False
    
    total_rewards_a = 0
    total_rewards_b = 0

    # --- 主迴圈 ---
    for ep in range(USER_CONFIG["test_episodes"]):
        print(f"\n--- 第 {ep + 1}/{USER_CONFIG['test_episodes']} 局開始 ---")
        obs_A_current, obs_B_current = env.reset()
        
        # 初始化 RNN 隱藏狀態 (如果需要)
        hidden_A = None
        if USER_CONFIG["model_a"]["type"] == "QNetRNN":
            assert isinstance(model_a, QNetRNN) # Type check
            hidden_A = model_a.init_hidden(1, device)

        hidden_B = None
        if USER_CONFIG["model_b"]["type"] == "QNetRNN":
            assert isinstance(model_b, QNetRNN) # Type check
            hidden_B = model_b.init_hidden(1, device)

        spin_angle_deg = 0.0  # 用於旋轉球圖片的角度 (度)
        ball_trail: List[Tuple[float, float]] = []
        
        ep_reward_a = 0
        ep_reward_b = 0
        done = False
        step_count = 0

        while not done:
            # --- 事件處理 ---
            if USER_CONFIG["enable_render"] and game_ui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env.close()
                        print("\n[信息] Pygame 關閉，程序結束。")
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                            print("[信息] 遊戲" + ("暫停" if paused else "繼續"))
                    if event.type == pygame.MOUSEBUTTONDOWN or \
                       (event.type == pygame.MOUSEMOTION and event.buttons[0]): # 按住左鍵拖動
                        game_ui.update_slider(event.pos, True)
            
            if paused and USER_CONFIG["enable_render"]:
                game_ui.draw(env, USER_CONFIG["model_a"]["name"], USER_CONFIG["model_b"]["name"], paused, ball_trail, spin_angle_deg)
                clock.tick(USER_CONFIG["render_fps"]) # 即使暫停也保持FPS限制
                continue

            # --- 選擇動作 ---
            action_A, hidden_A_next = select_action(obs_A_current, model_a, USER_CONFIG["model_a"]["type"], hidden_A, device)
            action_B, hidden_B_next = select_action(obs_B_current, model_b, USER_CONFIG["model_b"]["type"], hidden_B, device)
            
            if hidden_A_next: hidden_A = hidden_A_next
            if hidden_B_next: hidden_B = hidden_B_next

            # --- 環境交互 ---
            (obs_A_next, obs_B_next), (reward_A, reward_B), done, _ = env.step(action_A, action_B)
            
            ep_reward_a += reward_A
            ep_reward_b += reward_B

            obs_A_current = obs_A_next
            obs_B_current = obs_B_next
            step_count +=1

            # --- 更新視覺化信息 ---
            if USER_CONFIG["enable_render"] and game_ui:
                # 球的旋轉角度更新 (rad/step * 180/pi -> deg/step)
                spin_angle_deg += env.spin * (180.0 / math.pi)
                spin_angle_deg %= 360 # 保持在 0-360 度

                ball_pixel_x = int(env.ball_x * render_size)
                ball_pixel_y = int(env.ball_y * render_size)
                ball_trail.append((ball_pixel_x, ball_pixel_y))
                if len(ball_trail) > USER_CONFIG["trail_length"]:
                    ball_trail.pop(0)
                
                game_ui.draw(env, USER_CONFIG["model_a"]["name"], USER_CONFIG["model_b"]["name"], paused, ball_trail, spin_angle_deg)
                
                # 控制幀率
                clock.tick(USER_CONFIG["render_fps"] * game_ui.current_speed_factor if game_ui else USER_CONFIG["render_fps"])


            if done:
                print(f"局終: {USER_CONFIG['model_a']['name']} 得分 {env.scoreA}, {USER_CONFIG['model_b']['name']} 得分 {env.scoreB} (原始獎勵 A:{ep_reward_a:.2f}, B:{ep_reward_b:.2f})。步數: {step_count}")
                total_rewards_a += ep_reward_a
                total_rewards_b += ep_reward_b
                # 短暫停留顯示最後一幀結果
                if USER_CONFIG["enable_render"] and game_ui:
                    game_ui.draw(env, USER_CONFIG["model_a"]["name"], USER_CONFIG["model_b"]["name"], True, ball_trail, spin_angle_deg) # 顯示 paused=True
                    pygame.time.wait(1500) # 等待1.5秒

    # --- 所有測試局結束 ---
    print("\n\n--- 所有測試局結束 ---")
    print(f"總平均獎勵 {USER_CONFIG['model_a']['name']}: {total_rewards_a / USER_CONFIG['test_episodes']:.2f}")
    print(f"總平均獎勵 {USER_CONFIG['model_b']['name']}: {total_rewards_b / USER_CONFIG['test_episodes']:.2f}")

    env.close()
    if USER_CONFIG["enable_render"]:
        pygame.quit()
    print("[信息] 程序正常結束。")

if __name__ == "__main__":
    main()