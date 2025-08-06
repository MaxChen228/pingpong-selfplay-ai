"""
效果管理系統
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from ..config.constants import *


@dataclass
class Effect:
    """效果基類"""
    x: float
    y: float
    active: bool = True
    
    def update(self, dt: float = 1.0):
        """更新效果"""
        pass
    
    def is_alive(self) -> bool:
        """檢查效果是否還活著"""
        return self.active


@dataclass
class CollisionEffect(Effect):
    """碰撞效果"""
    radius: float = COLLISION_EFFECT_RADIUS_INIT
    alpha: int = 255
    color: Tuple[int, int, int] = COLLISION_EFFECT_COLOR
    
    def update(self, dt: float = 1.0):
        """更新碰撞效果"""
        self.radius += COLLISION_EFFECT_RADIUS_GROW * dt
        self.alpha -= COLLISION_EFFECT_ALPHA_DECAY * dt
        
        if self.alpha <= 0:
            self.alpha = 0
            self.active = False
    
    def render_data(self) -> Dict:
        """獲取渲染數據"""
        return {
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'alpha': int(self.alpha),
            'color': self.color
        }


@dataclass
class ParticleEffect(Effect):
    """粒子效果（為第二階段準備）"""
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    lifetime: float = 1.0
    size: float = 2.0
    color: Tuple[int, int, int] = (255, 255, 255)
    
    def update(self, dt: float = 1.0):
        """更新粒子"""
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        self.lifetime -= dt
        
        if self.lifetime <= 0:
            self.active = False


class EffectManager:
    """效果管理器"""
    
    def __init__(self):
        self.effects: List[Effect] = []
        self.collision_effects: List[CollisionEffect] = []
        self.particle_effects: List[ParticleEffect] = []
    
    def add_collision(self, x: float, y: float):
        """添加碰撞效果"""
        effect = CollisionEffect(x=x, y=y)
        self.collision_effects.append(effect)
        self.effects.append(effect)
    
    def add_particle(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """添加粒子效果"""
        effect = ParticleEffect(x=x, y=y, velocity_x=vx, velocity_y=vy)
        self.particle_effects.append(effect)
        self.effects.append(effect)
    
    def update(self, dt: float = 1.0):
        """更新所有效果"""
        # 更新效果
        for effect in self.effects:
            effect.update(dt)
        
        # 清理死亡的效果
        self.effects = [e for e in self.effects if e.is_alive()]
        self.collision_effects = [e for e in self.collision_effects if e.is_alive()]
        self.particle_effects = [e for e in self.particle_effects if e.is_alive()]
    
    def get_collision_render_data(self) -> List[Dict]:
        """獲取碰撞效果的渲染數據"""
        return [effect.render_data() for effect in self.collision_effects]
    
    def clear(self):
        """清空所有效果"""
        self.effects.clear()
        self.collision_effects.clear()
        self.particle_effects.clear()
    
    def active_count(self) -> Dict[str, int]:
        """獲取活躍效果數量"""
        return {
            'total': len(self.effects),
            'collisions': len(self.collision_effects),
            'particles': len(self.particle_effects)
        }