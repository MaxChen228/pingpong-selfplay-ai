"""
抽象渲染器接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Renderer(ABC):
    """渲染器抽象基類"""
    
    @abstractmethod
    def init(self, width: int, height: int, title: str = ""):
        """初始化渲染器"""
        pass
    
    @abstractmethod
    def clear(self):
        """清空畫面"""
        pass
    
    @abstractmethod
    def draw_background(self):
        """繪製背景"""
        pass
    
    @abstractmethod
    def draw_paddle(self, x: int, y: int, width: int, height: int, 
                   color: Tuple[int, int, int], is_top: bool = True):
        """繪製擋板"""
        pass
    
    @abstractmethod
    def draw_ball(self, x: int, y: int, radius: int, 
                 speed: float, spin_angle: float):
        """繪製球"""
        pass
    
    @abstractmethod
    def draw_trail(self, points: List[Tuple[int, int]]):
        """繪製軌跡"""
        pass
    
    @abstractmethod
    def draw_text(self, text: str, x: int, y: int, 
                 size: int = 20, color: Tuple[int, int, int] = (255, 255, 255)):
        """繪製文字"""
        pass
    
    @abstractmethod
    def draw_panel(self, x: int, y: int, width: int, height: int, 
                  alpha: int = 180):
        """繪製面板"""
        pass
    
    @abstractmethod
    def present(self):
        """呈現畫面"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """清理資源"""
        pass
    
    @abstractmethod
    def handle_events(self) -> dict:
        """處理事件"""
        pass