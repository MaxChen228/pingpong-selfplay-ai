"""
碰撞檢測系統
"""

from typing import Tuple, Optional
from ..config.constants import COLLISION_THRESHOLD_TOP, COLLISION_THRESHOLD_BOTTOM


class CollisionDetector:
    """碰撞檢測器"""
    
    @staticmethod
    def check_paddle_collision(ball_x: float, ball_y: float, 
                              last_ball_y: float,
                              paddle_x: float, paddle_width: float,
                              is_top: bool = True) -> bool:
        """
        檢測球是否與擋板碰撞
        
        Args:
            ball_x: 球的x座標
            ball_y: 球的y座標
            last_ball_y: 上一幀球的y座標
            paddle_x: 擋板中心x座標
            paddle_width: 擋板寬度
            is_top: 是否是上方擋板
            
        Returns:
            是否發生碰撞
        """
        if is_top:
            # 檢測上擋板碰撞
            if last_ball_y > COLLISION_THRESHOLD_TOP and ball_y <= COLLISION_THRESHOLD_TOP:
                return abs(ball_x - paddle_x) < paddle_width / 2
        else:
            # 檢測下擋板碰撞
            if last_ball_y < COLLISION_THRESHOLD_BOTTOM and ball_y >= COLLISION_THRESHOLD_BOTTOM:
                return abs(ball_x - paddle_x) < paddle_width / 2
        
        return False
    
    @staticmethod
    def get_collision_point(ball_x: float, is_top: bool = True) -> Tuple[float, float]:
        """
        獲取碰撞點座標
        
        Args:
            ball_x: 球的x座標
            is_top: 是否是上方擋板
            
        Returns:
            碰撞點的(x, y)座標
        """
        y = COLLISION_THRESHOLD_TOP if is_top else COLLISION_THRESHOLD_BOTTOM
        return (ball_x, y)