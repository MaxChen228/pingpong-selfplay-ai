#!/usr/bin/env python3
"""
Pong Viewer V2 - 重構版本
採用模組化架構，提高可維護性和擴展性
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional
import yaml
import torch
import math

# 添加專案根目錄
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pingpong_viewer.config import Settings, load_settings, constants
from pingpong_viewer.core import GameState, CollisionDetector
from pingpong_viewer.models import ModelLoader, Agent, AgentPair
from pingpong_viewer.rendering import PygameRenderer, EffectManager

# 環境導入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.my_pong_env_2p import PongEnv2P


class PongViewer:
    """主視覺化應用"""
    
    def __init__(self, settings: Settings):
        """初始化視覺化器"""
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化組件
        self.env = None
        self.agents = None
        self.renderer = None
        self.effect_manager = EffectManager()
        self.game_state = GameState()
        self.collision_detector = CollisionDetector()
        self.model_loader = ModelLoader(self.device)
        
        # 環境參數
        self.env_params = {}
        self.rnn_arch_config = {}
        
    def initialize(self):
        """初始化系統"""
        print(f"[信息] 使用設備: {self.device}")
        
        # 載入配置
        self._load_configs()
        
        # 創建環境
        self._create_environment()
        
        # 載入模型
        self._load_models()
        
        # 初始化渲染器
        if self.settings.enable_render:
            self._init_renderer()
    
    def _load_configs(self):
        """載入配置文件"""
        # 環境配置
        with open(self.settings.env_config_path, "r") as f:
            self.env_params = yaml.safe_load(f)["env"]
            self.env_params["enable_render"] = False
        
        # RNN配置
        rnn_path = Path(self.settings.rnn_model_config_path)
        if rnn_path.exists():
            with open(rnn_path, "r") as f:
                self.rnn_arch_config = yaml.safe_load(f).get("training", {})
    
    def _create_environment(self):
        """創建環境"""
        self.env = PongEnv2P(**self.env_params)
        self.render_size = self.env_params.get("render_size", constants.DEFAULT_RENDER_SIZE)
    
    def _load_models(self):
        """載入AI模型"""
        print(f"[信息] 載入模型 A ({self.settings.model_a['type']}): {self.settings.model_a['path']}")
        model_a = self.model_loader.load(self.settings.model_a, self.rnn_arch_config)
        
        print(f"[信息] 載入模型 B ({self.settings.model_b['type']}): {self.settings.model_b['path']}")
        model_b = self.model_loader.load(self.settings.model_b, self.rnn_arch_config)
        
        # 創建代理
        agent_a = Agent(model_a, self.settings.model_a['type'],
                       self.settings.model_a['name'], 
                       self.settings.model_a['color'],
                       self.device)
        
        agent_b = Agent(model_b, self.settings.model_b['type'],
                       self.settings.model_b['name'],
                       self.settings.model_b['color'],
                       self.device)
        
        self.agents = AgentPair(agent_a, agent_b)
    
    def _init_renderer(self):
        """初始化渲染器"""
        self.renderer = PygameRenderer()
        title = f"Pong Viewer: {self.agents.agent_a.name} vs {self.agents.agent_b.name}"
        self.renderer.init(self.render_size, self.render_size, title)
        
        # 載入球圖片
        if self.settings.ball_image_path:
            ball_diameter = int(0.03 * 2 * self.render_size * 1.5)
            self.renderer.load_ball_image(self.settings.ball_image_path, ball_diameter)
    
    def run(self):
        """運行視覺化"""
        self.game_state.total_episodes = self.settings.test_episodes
        
        for episode in range(self.settings.test_episodes):
            self.game_state.current_episode = episode + 1
            print(f"\n--- 第 {episode + 1}/{self.settings.test_episodes} 局開始 ---")
            
            self._run_episode()
            
            # 顯示統計
            stats = self.agents.get_stats()
            print(f"局終: {self.agents.agent_a.name} 得分 {self.env.scoreA}, "
                  f"{self.agents.agent_b.name} 得分 {self.env.scoreB}")
        
        self._show_final_stats()
        self.cleanup()
    
    def _run_episode(self):
        """運行單個回合"""
        # 重置狀態
        obs_a, obs_b = self.env.reset()
        self.agents.reset_episode()
        self.game_state.reset_episode()
        self.game_state.last_ball_y = self.env.ball_y
        self.effect_manager.clear()
        
        while not self.game_state.done:
            # 處理事件
            if self.settings.enable_render:
                if not self._handle_events():
                    return  # 退出
            
            # 暫停處理
            if self.game_state.paused:
                if self.settings.enable_render:
                    self._render_frame()
                    self.renderer.tick(self.settings.render_fps)
                continue
            
            # AI決策
            action_a, action_b = self.agents.select_actions(obs_a, obs_b)
            
            # 環境步進
            (obs_a, obs_b), (reward_a, reward_b), done, _ = self.env.step(action_a, action_b)
            
            # 更新狀態
            self.game_state.done = done
            self.game_state.increment_step()
            self.agents.update_rewards(reward_a, reward_b)
            
            # 碰撞檢測
            if self.settings.enable_effects:
                self._check_collisions()
            
            # 更新視覺狀態
            self.game_state.last_ball_y = self.env.ball_y
            self.game_state.update_spin_angle(self.env.spin)
            self.game_state.update_ball_trail(self.env.ball_x, self.env.ball_y, self.render_size)
            
            # 渲染
            if self.settings.enable_render:
                self._render_frame()
                fps = self.game_state.get_effective_fps(self.settings.render_fps)
                self.renderer.tick(fps)
        
        # 記錄結果
        self.agents.record_result(self.env.scoreA, self.env.scoreB)
        
        # 顯示獲勝動畫
        if self.settings.enable_render:
            self._show_winner()
    
    def _handle_events(self) -> bool:
        """處理事件，返回是否繼續"""
        events = self.renderer.handle_events()
        
        if events['quit']:
            return False
        
        if events['space']:
            self.game_state.toggle_pause()
            print("[信息] 遊戲" + ("暫停" if self.game_state.paused else "繼續"))
        
        # 處理滑桿
        if events['mouse_pressed']:
            self._handle_slider(events['mouse_pos'])
        
        return True
    
    def _handle_slider(self, mouse_pos: Tuple[int, int]):
        """處理滑桿交互"""
        # 簡化的滑桿處理
        slider_x = 50
        slider_y = self.render_size - constants.SLIDER_BOTTOM_MARGIN
        slider_width = constants.SLIDER_WIDTH
        
        if slider_x <= mouse_pos[0] <= slider_x + slider_width:
            progress = (mouse_pos[0] - slider_x) / slider_width
            speed = constants.SPEED_MIN + (constants.SPEED_MAX - constants.SPEED_MIN) * progress
            self.game_state.set_speed_factor(speed)
    
    def _check_collisions(self):
        """檢測碰撞"""
        # 上擋板碰撞
        if self.collision_detector.check_paddle_collision(
            self.env.ball_x, self.env.ball_y, self.game_state.last_ball_y,
            self.env.top_paddle_x, self.env.paddle_width, is_top=True
        ):
            x, y = self.collision_detector.get_collision_point(self.env.ball_x, is_top=True)
            self.effect_manager.add_collision(x, y)
        
        # 下擋板碰撞
        if self.collision_detector.check_paddle_collision(
            self.env.ball_x, self.env.ball_y, self.game_state.last_ball_y,
            self.env.bottom_paddle_x, self.env.paddle_width, is_top=False
        ):
            x, y = self.collision_detector.get_collision_point(self.env.ball_x, is_top=False)
            self.effect_manager.add_collision(x, y)
    
    def _render_frame(self):
        """渲染一幀"""
        # 背景
        self.renderer.draw_background()
        
        # 更新效果
        self.effect_manager.update()
        
        # 繪製碰撞效果
        for effect_data in self.effect_manager.get_collision_render_data():
            self._render_collision_effect(effect_data)
        
        # 繪製軌跡
        self.renderer.draw_trail(self.game_state.ball_trail)
        
        # 繪製擋板
        paddle_width = int(self.env.paddle_width * self.render_size)
        
        # 上擋板
        top_x = int(self.env.top_paddle_x * self.render_size)
        self.renderer.draw_paddle(top_x, 0, paddle_width, constants.PADDLE_HEIGHT,
                                 self.agents.agent_a.color, is_top=True)
        
        # 下擋板
        bottom_x = int(self.env.bottom_paddle_x * self.render_size)
        self.renderer.draw_paddle(bottom_x, self.render_size - constants.PADDLE_HEIGHT,
                                 paddle_width, constants.PADDLE_HEIGHT,
                                 self.agents.agent_b.color, is_top=False)
        
        # 繪製球
        ball_x = int(self.env.ball_x * self.render_size)
        ball_y = int(self.env.ball_y * self.render_size)
        ball_radius = int(self.env.world_ball_radius * self.render_size)
        ball_speed = math.hypot(self.env.ball_vx, self.env.ball_vy)
        
        self.renderer.draw_ball(ball_x, ball_y, ball_radius, 
                               ball_speed, self.game_state.spin_angle_deg)
        
        # 繪製UI
        self._render_ui()
        
        # 呈現
        self.renderer.present()
    
    def _render_collision_effect(self, effect_data: dict):
        """渲染碰撞效果"""
        import pygame
        
        if effect_data['alpha'] > 0:
            effect_surface = pygame.Surface((effect_data['radius'] * 2, effect_data['radius'] * 2), 
                                           pygame.SRCALPHA)
            color_with_alpha = (*effect_data['color'], effect_data['alpha'])
            pygame.draw.circle(effect_surface, color_with_alpha,
                             (int(effect_data['radius']), int(effect_data['radius'])), 
                             int(effect_data['radius']))
            
            x = int(effect_data['x'] * self.render_size - effect_data['radius'])
            y = int(effect_data['y'] * self.render_size - effect_data['radius'])
            self.renderer.screen.blit(effect_surface, (x, y))
    
    def _render_ui(self):
        """渲染UI元素"""
        # 信息面板
        self.renderer.draw_panel(0, 0, self.render_size, 90)
        
        # 分數
        score_text = f"{self.env.scoreA} - {self.env.scoreB}"
        self.renderer.draw_text(score_text, self.render_size // 2, 25, 
                               size='large', center=True)
        
        # 模型名稱
        self.renderer.draw_text(self.agents.agent_a.name, 10, 45, size='medium',
                               color=self.agents.agent_a.color)
        
        # 速度信息
        ball_speed = math.hypot(self.env.ball_vx, self.env.ball_vy)
        speed_text = f"Speed: {ball_speed:.3f}"
        self.renderer.draw_text(speed_text, 10, 70, size='small',
                               color=constants.THEME_COLORS['text_secondary'])
        
        # 旋轉信息
        spin_text = f"Spin: {self.env.spin:+.2f}"
        self.renderer.draw_text(spin_text, 150, 70, size='small',
                               color=constants.THEME_COLORS['text_secondary'])
        
        # 滑桿
        slider_value = self.renderer.draw_slider(
            50, self.render_size - constants.SLIDER_BOTTOM_MARGIN,
            constants.SLIDER_WIDTH, constants.SLIDER_HEIGHT,
            self.game_state.speed_factor, constants.SPEED_MIN, constants.SPEED_MAX
        )
        
        # 速度文字
        speed_factor_text = f"Speed x{self.game_state.speed_factor:.1f}"
        self.renderer.draw_text(speed_factor_text, 270, 
                               self.render_size - constants.SLIDER_BOTTOM_MARGIN,
                               size='medium')
        
        # 暫停提示
        if self.game_state.paused:
            self.renderer.draw_text("PAUSED", self.render_size // 2, 
                                   self.render_size // 2, 
                                   size='large', color=constants.THEME_COLORS['text_warning'],
                                   center=True)
            self.renderer.draw_text("Press SPACE to continue", 
                                   self.render_size // 2,
                                   self.render_size // 2 + constants.PAUSE_HINT_OFFSET,
                                   size='small', center=True)
    
    def _show_winner(self):
        """顯示獲勝者"""
        if not self.renderer:
            return
        
        winner_name = (self.agents.agent_a.name if self.env.scoreA > self.env.scoreB 
                      else self.agents.agent_b.name)
        winner_color = (self.agents.agent_a.color if self.env.scoreA > self.env.scoreB
                       else self.agents.agent_b.color)
        
        # 繪製最後一幀
        self._render_frame()
        
        # 獲勝文字
        winner_text = f"{winner_name} WINS!"
        self.renderer.draw_text(winner_text, self.render_size // 2,
                               self.render_size // 2 - 50,
                               size='large', color=winner_color, center=True)
        
        self.renderer.present()
        
        # 等待
        import pygame
        pygame.time.wait(constants.WIN_DISPLAY_DURATION)
    
    def _show_final_stats(self):
        """顯示最終統計"""
        print("\n\n--- 所有測試局結束 ---")
        stats = self.agents.get_stats()
        
        print(f"總平均獎勵 {self.agents.agent_a.name}: "
              f"{stats['agent_a']['average_reward']:.2f}")
        print(f"總平均獎勵 {self.agents.agent_b.name}: "
              f"{stats['agent_b']['average_reward']:.2f}")
    
    def cleanup(self):
        """清理資源"""
        if self.env:
            self.env.close()
        if self.renderer:
            self.renderer.cleanup()
        print("[信息] 程序正常結束。")


def main():
    """主函數"""
    # 載入配置
    settings = load_settings()
    
    # 創建並運行視覺化器
    viewer = PongViewer(settings)
    viewer.initialize()
    viewer.run()


if __name__ == "__main__":
    main()