"""
模型載入器
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.qnet import QNet
from models.qnet_rnn import QNetRNN


class ModelLoader:
    """統一的模型載入器"""
    
    # 模型類型映射
    MODEL_CLASSES = {
        'QNet': QNet,
        'QNetRNN': QNetRNN,
    }
    
    # 常見的state dict鍵名
    STATE_DICT_KEYS = [
        "modelB_state", "modelA_state", "modelB", "modelA", 
        "model", "state_dict"
    ]
    
    def __init__(self, device: Optional[torch.device] = None):
        """初始化載入器"""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 7  # PongEnv2P 觀察空間維度
        self.output_dim = 3  # PongEnv2P 動作空間維度
    
    def load(self, model_config: Dict[str, Any], 
             rnn_arch_config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        載入模型
        
        Args:
            model_config: 包含 'path', 'type', 'name' 的字典
            rnn_arch_config: RNN架構配置（如果需要）
        
        Returns:
            載入的模型
        """
        model_path = Path(model_config['path'])
        model_type = model_config['type']
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 載入checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 提取state dict
        state_dict = self._extract_state_dict(checkpoint, model_path)
        
        # 創建並載入模型
        if model_type == 'QNet':
            model = self._load_qnet(state_dict, model_path)
        elif model_type == 'QNetRNN':
            model = self._load_qnet_rnn(state_dict, rnn_arch_config)
        else:
            raise ValueError(f"不支持的模型類型: {model_type}")
        
        # 設置為評估模式
        model.eval()
        
        # 重置噪聲（如果有）
        if hasattr(model, 'reset_noise'):
            model.reset_noise()
        
        return model
    
    def _extract_state_dict(self, checkpoint: Dict, model_path: Path) -> Dict:
        """從checkpoint提取state dict"""
        # 嘗試常見的鍵名
        for key in self.STATE_DICT_KEYS:
            if key in checkpoint:
                return checkpoint[key]
        
        # 檢查是否是舊格式（直接是state_dict）
        if self._is_old_qnet_format(checkpoint):
            return checkpoint
        
        raise KeyError(
            f"在 checkpoint '{model_path}' 中找不到有效的模型狀態字典。"
            f"已嘗試的鍵名: {self.STATE_DICT_KEYS}"
        )
    
    def _is_old_qnet_format(self, checkpoint: Dict) -> bool:
        """檢查是否是舊的QNet格式"""
        if not all(not isinstance(v, dict) for v in checkpoint.values()):
            return False
        
        return any(k.startswith(("fc.", "features.")) for k in checkpoint.keys())
    
    def _load_qnet(self, state_dict: Dict, model_path: Path) -> QNet:
        """載入QNet模型"""
        model = QNet(input_dim=self.input_dim, output_dim=self.output_dim).to(self.device)
        
        # 檢查格式
        is_new_format = any(k.startswith(("features.", "fc_V.", "fc_A.")) 
                           for k in state_dict)
        
        if is_new_format:
            model.load_state_dict(state_dict, strict=True)
        else:
            # 轉換舊格式
            mapped_state = self._map_old_qnet_state(state_dict)
            model.load_state_dict(mapped_state, strict=False)
            print(f"[信息] 已將舊式 QNet checkpoint '{model_path}' 映射到新架構")
        
        return model
    
    def _map_old_qnet_state(self, state_dict: Dict) -> Dict:
        """映射舊的QNet state dict到新格式"""
        mapped = {}
        
        # 映射特徵層
        for k, v in state_dict.items():
            if k.startswith("fc.0."):
                mapped[k.replace("fc.0.", "features.0.")] = v
            elif k.startswith("fc.2."):
                mapped[k.replace("fc.2.", "features.2.")] = v
        
        # 映射最後一層到Dueling Heads
        if "fc.4.weight" in state_dict and "fc.4.bias" in state_dict:
            w4 = state_dict["fc.4.weight"]
            b4 = state_dict["fc.4.bias"]
            mapped["fc_A.weight_mu"] = w4
            mapped["fc_A.bias_mu"] = b4
            mapped["fc_V.weight_mu"] = w4.mean(dim=0, keepdim=True)
            mapped["fc_V.bias_mu"] = b4.mean().unsqueeze(0)
        
        return mapped
    
    def _load_qnet_rnn(self, state_dict: Dict, 
                      rnn_arch_config: Optional[Dict[str, Any]]) -> QNetRNN:
        """載入QNetRNN模型"""
        if rnn_arch_config is None:
            rnn_arch_config = {}
        
        model = QNetRNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            feature_dim=rnn_arch_config.get('feature_dim', 128),
            lstm_hidden_dim=rnn_arch_config.get('lstm_hidden_dim', 128),
            lstm_layers=rnn_arch_config.get('lstm_layers', 1),
            head_hidden_dim=rnn_arch_config.get('head_hidden_dim', 128)
        ).to(self.device)
        
        model.load_state_dict(state_dict)
        return model