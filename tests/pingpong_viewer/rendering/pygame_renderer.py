"""
Pygame渲染器實現
"""

import pygame
import pygame.gfxdraw
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .renderer import Renderer
from ..config.constants import *


class PygameRenderer(Renderer):
    """Pygame渲染器"""
    
    def __init__(self):
        self.screen = None
        self.clock = None
        self.fonts = {}
        self.width = 0
        self.height = 0
        self.grid_surface = None
        self.ball_image = None
        
    def init(self, width: int, height: int, title: str = ""):
        """初始化Pygame"""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        
        # 初始化字體
        self._init_fonts()
        
        # 創建網格背景
        self.grid_surface = self._create_grid_background()
        
    def _init_fonts(self):
        """初始化字體"""
        try:
            self.fonts['small'] = pygame.font.SysFont(FONT_FAMILY_PRIMARY, FONT_SIZE_SMALL)
            self.fonts['medium'] = pygame.font.SysFont(FONT_FAMILY_PRIMARY, FONT_SIZE_MEDIUM)
            self.fonts['large'] = pygame.font.SysFont(FONT_FAMILY_PRIMARY, FONT_SIZE_LARGE)
        except:
            self.fonts['small'] = pygame.font.SysFont(None, FONT_SIZE_SMALL)
            self.fonts['medium'] = pygame.font.SysFont(None, FONT_SIZE_MEDIUM)
            self.fonts['large'] = pygame.font.SysFont(None, FONT_SIZE_LARGE)
    
    def _create_grid_background(self) -> pygame.Surface:
        """創建網格背景"""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(THEME_COLORS['background'])
        
        # 繪製網格線
        for x in range(0, self.width, GRID_SPACING):
            pygame.draw.line(surface, THEME_COLORS['grid'], 
                           (x, 0), (x, self.height), 1)
        for y in range(0, self.height, GRID_SPACING):
            pygame.draw.line(surface, THEME_COLORS['grid'], 
                           (0, y), (self.width, y), 1)
        
        # 中心線
        pygame.draw.line(surface, THEME_COLORS['grid_center'], 
                        (0, self.height // 2), (self.width, self.height // 2), 2)
        
        return surface
    
    def load_ball_image(self, image_path: str, diameter: int):
        """載入球圖片"""
        if image_path and Path(image_path).exists():
            try:
                self.ball_image = pygame.image.load(image_path).convert_alpha()
                self.ball_image = pygame.transform.scale(self.ball_image, (diameter, diameter))
            except pygame.error as e:
                print(f"無法載入球圖片: {e}")
                self.ball_image = None
    
    def clear(self):
        """清空畫面"""
        self.screen.fill(THEME_COLORS['background'])
    
    def draw_background(self):
        """繪製背景"""
        if self.grid_surface:
            self.screen.blit(self.grid_surface, (0, 0))
    
    def draw_paddle(self, x: int, y: int, width: int, height: int,
                   color: Tuple[int, int, int], is_top: bool = True):
        """繪製帶發光效果的擋板"""
        # 發光效果
        glow_surface = pygame.Surface((width + PADDLE_GLOW_WIDTH_OFFSET, 
                                      height + PADDLE_GLOW_HEIGHT_OFFSET), 
                                     pygame.SRCALPHA)
        glow_color = (*color, GLOW_ALPHA_BASE)
        
        for i in range(GLOW_LAYERS):
            glow_rect = pygame.Rect(20 - i*5, 10 - i*3, 
                                   width + i*10, height + i*6)
            pygame.draw.rect(glow_surface, glow_color, glow_rect, border_radius=5)
        
        glow_pos = (x - width // 2 - 20, y - 10)
        self.screen.blit(glow_surface, glow_pos)
        
        # 主擋板漸層
        paddle_rect = pygame.Rect(x - width // 2, y, width, height)
        for i in range(height):
            progress = i / height
            if is_top:
                progress = 1 - progress
            gradient_color = tuple(int(c * (0.7 + 0.3 * progress)) for c in color)
            pygame.draw.line(self.screen, gradient_color,
                           (paddle_rect.left, paddle_rect.top + i),
                           (paddle_rect.right, paddle_rect.top + i))
        
        # 邊框
        pygame.draw.rect(self.screen, (255, 255, 255), paddle_rect, 2, border_radius=3)
    
    def draw_ball(self, x: int, y: int, radius: int,
                 speed: float, spin_angle: float):
        """繪製帶光暈的球"""
        # 速度顏色
        speed_factor = min(speed / BALL_SPEED_COLOR_FACTOR, 1.0)
        ball_color = (
            255,
            int(255 * (1 - speed_factor * 0.3)),
            int(255 * (1 - speed_factor * 0.5))
        )
        
        # 光暈
        glow_radius = radius + BALL_GLOW_RADIUS_OFFSET
        for i in range(5):
            alpha = 30 - i * GLOW_ALPHA_STEP
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), 
                                         pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*ball_color, alpha),
                             (glow_radius, glow_radius), glow_radius - i * 2)
            self.screen.blit(glow_surface, (x - glow_radius, y - glow_radius))
        
        # 球體
        if self.ball_image:
            rotated = pygame.transform.rotate(self.ball_image, -spin_angle)
            rect = rotated.get_rect(center=(x, y))
            self.screen.blit(rotated, rect)
        else:
            pygame.draw.circle(self.screen, ball_color, (x, y), radius)
            # 高光
            highlight_pos = (x - radius // BALL_HIGHLIGHT_OFFSET_RATIO,
                           y - radius // BALL_HIGHLIGHT_OFFSET_RATIO)
            pygame.draw.circle(self.screen, (255, 255, 255), highlight_pos,
                             radius // BALL_HIGHLIGHT_SIZE_RATIO)
    
    def draw_trail(self, points: List[Tuple[int, int]]):
        """繪製漸變軌跡"""
        if len(points) < 2:
            return
        
        trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for i in range(1, len(points)):
            alpha = int((i / len(points)) * 150)
            width = max(1, int((i / len(points)) * 3))
            color = (200, 200, 255, alpha)
            pygame.draw.line(trail_surface, color, points[i-1], points[i], width)
        
        self.screen.blit(trail_surface, (0, 0))
    
    def draw_text(self, text: str, x: int, y: int,
                 size: str = 'medium', color: Tuple[int, int, int] = None,
                 center: bool = False):
        """繪製文字"""
        if color is None:
            color = THEME_COLORS['text_primary']
        
        font = self.fonts.get(size, self.fonts['medium'])
        surface = font.render(text, True, color)
        
        if center:
            rect = surface.get_rect(center=(x, y))
            self.screen.blit(surface, rect)
        else:
            self.screen.blit(surface, (x, y))
    
    def draw_panel(self, x: int, y: int, width: int, height: int,
                  alpha: int = 180):
        """繪製半透明面板"""
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, alpha))
        self.screen.blit(panel, (x, y))
    
    def draw_slider(self, x: int, y: int, width: int, height: int,
                   value: float, min_val: float, max_val: float) -> Dict:
        """繪製滑桿"""
        # 背景
        slider_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (50, 50, 50), slider_rect, border_radius=6)
        pygame.draw.rect(self.screen, (100, 100, 100), slider_rect, 2, border_radius=6)
        
        # 進度
        progress = (value - min_val) / (max_val - min_val)
        fill_width = int(width * progress)
        fill_rect = pygame.Rect(x, y, fill_width, height)
        pygame.draw.rect(self.screen, (100, 150, 255), fill_rect, border_radius=6)
        
        # 滑塊
        knob_x = x + int((width - SLIDER_KNOB_WIDTH) * progress)
        knob_y = y - (SLIDER_KNOB_HEIGHT - height) // 2
        knob_rect = pygame.Rect(knob_x, knob_y, SLIDER_KNOB_WIDTH, SLIDER_KNOB_HEIGHT)
        
        # 陰影
        shadow_rect = knob_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        pygame.draw.rect(self.screen, (30, 30, 30), shadow_rect, border_radius=8)
        
        # 主體
        pygame.draw.rect(self.screen, (255, 255, 255), knob_rect, border_radius=8)
        pygame.draw.rect(self.screen, (150, 150, 150), knob_rect, 2, border_radius=8)
        
        return {'rect': slider_rect, 'knob_rect': knob_rect, 'value': value}
    
    def present(self):
        """呈現畫面"""
        pygame.display.flip()
    
    def cleanup(self):
        """清理資源"""
        pygame.quit()
    
    def handle_events(self) -> Dict:
        """處理事件"""
        events = {
            'quit': False,
            'space': False,
            'mouse_pos': pygame.mouse.get_pos(),
            'mouse_pressed': pygame.mouse.get_pressed()[0],
            'keys': []
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    events['space'] = True
                events['keys'].append(event.key)
        
        return events
    
    def tick(self, fps: float):
        """控制幀率"""
        if self.clock:
            self.clock.tick(fps)