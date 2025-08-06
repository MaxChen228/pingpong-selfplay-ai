"""
遊戲狀態管理
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class GameState:
    """遊戲狀態"""
    
    # 遊戲控制
    paused: bool = False
    done: bool = False
    current_episode: int = 0
    total_episodes: int = 0
    
    # 速度控制
    speed_factor: float = 1.0
    
    # 球軌跡
    ball_trail: List[Tuple[float, float]] = field(default_factory=list)
    max_trail_length: int = 50
    
    # 球旋轉角度
    spin_angle_deg: float = 0.0
    
    # 碰撞檢測
    last_ball_y: float = 0.5
    
    # 計步器
    step_count: int = 0
    
    def reset_episode(self):
        """重置回合狀態"""
        self.done = False
        self.ball_trail.clear()
        self.spin_angle_deg = 0.0
        self.last_ball_y = 0.5
        self.step_count = 0
    
    def update_ball_trail(self, x: float, y: float, render_size: int):
        """更新球軌跡"""
        pixel_x = int(x * render_size)
        pixel_y = int(y * render_size)
        self.ball_trail.append((pixel_x, pixel_y))
        
        # 限制軌跡長度
        if len(self.ball_trail) > self.max_trail_length:
            self.ball_trail.pop(0)
    
    def update_spin_angle(self, spin: float):
        """更新旋轉角度"""
        self.spin_angle_deg += spin * (180.0 / math.pi)
        self.spin_angle_deg %= 360
    
    def toggle_pause(self):
        """切換暫停狀態"""
        self.paused = not self.paused
    
    def increment_step(self):
        """增加步數"""
        self.step_count += 1
    
    def set_speed_factor(self, factor: float):
        """設置速度因子"""
        self.speed_factor = max(0.1, min(5.0, factor))
    
    def is_playing(self) -> bool:
        """是否正在遊戲中"""
        return not self.done and not self.paused
    
    def get_effective_fps(self, base_fps: int) -> float:
        """獲取有效FPS"""
        return base_fps * self.speed_factor