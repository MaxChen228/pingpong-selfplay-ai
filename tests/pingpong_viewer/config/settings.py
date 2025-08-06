"""
配置管理系統
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import json


class Settings:
    """配置管理類"""
    
    def __init__(self):
        # 環境配置
        self.env_config_path = "config.yaml"
        self.rnn_model_config_path = "config_rnn.yaml"
        
        # 模型配置
        self.model_a = {
            "name": "Model_A_QNet",
            "path": "checkpoints/model4-0.pth",
            "type": "QNet",
            "color": (100, 200, 255),  # 淺藍
        }
        
        self.model_b = {
            "name": "RNN_Gen6",
            "path": "checkpoints_rnn/rnn_pong_soul_2.pth",
            "type": "QNetRNN",
            "color": (255, 150, 100),  # 橘色
        }
        
        # 遊戲設定
        self.test_episodes = 2
        self.ball_image_path = "assets/sunglasses.png"
        self.render_fps = 60
        self.trail_length = 50
        self.enable_render = True
        self.enable_effects = True
        
    def load_from_file(self, config_path: str):
        """從文件載入配置"""
        path = Path(config_path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"不支援的配置文件格式: {path.suffix}")
        
        # 更新配置
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config = {
            'env_config_path': self.env_config_path,
            'rnn_model_config_path': self.rnn_model_config_path,
            'model_a': self.model_a,
            'model_b': self.model_b,
            'test_episodes': self.test_episodes,
            'ball_image_path': self.ball_image_path,
            'render_fps': self.render_fps,
            'trail_length': self.trail_length,
            'enable_render': self.enable_render,
            'enable_effects': self.enable_effects,
        }
        
        path = Path(config_path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> bool:
        """驗證配置的有效性"""
        # 檢查模型路徑
        for model_key in ['model_a', 'model_b']:
            model = getattr(self, model_key)
            if not Path(model['path']).exists():
                print(f"警告: 模型文件不存在: {model['path']}")
        
        # 檢查球圖片
        if self.ball_image_path and not Path(self.ball_image_path).exists():
            print(f"警告: 球圖片不存在: {self.ball_image_path}")
            self.ball_image_path = None
        
        return True


def load_settings(config_path: str = None) -> Settings:
    """載入配置的便捷函數"""
    settings = Settings()
    
    if config_path and Path(config_path).exists():
        settings.load_from_file(config_path)
    
    settings.validate()
    return settings