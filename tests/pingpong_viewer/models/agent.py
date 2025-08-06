"""
AI代理封裝
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np


class Agent:
    """AI代理"""
    
    def __init__(self, model: nn.Module, model_type: str, 
                 name: str, color: Tuple[int, int, int],
                 device: torch.device):
        """
        初始化代理
        
        Args:
            model: 神經網路模型
            model_type: 模型類型 ('QNet' 或 'QNetRNN')
            name: 代理名稱
            color: 代理顏色
            device: 計算設備
        """
        self.model = model
        self.model_type = model_type
        self.name = name
        self.color = color
        self.device = device
        
        # RNN狀態
        self.hidden_state = None
        self.needs_hidden = (model_type == 'QNetRNN')
        
        # 統計數據
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.wins = 0
        self.losses = 0
    
    def reset_episode(self):
        """重置回合狀態"""
        self.episode_reward = 0.0
        
        # 初始化RNN隱藏狀態
        if self.needs_hidden:
            self.hidden_state = self._init_hidden_state()
    
    def _init_hidden_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化RNN隱藏狀態"""
        if hasattr(self.model, 'init_hidden'):
            return self.model.init_hidden(1, self.device)
        else:
            # 默認初始化
            hidden_dim = 128  # 默認值
            return (torch.zeros(1, 1, hidden_dim, device=self.device),
                   torch.zeros(1, 1, hidden_dim, device=self.device))
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        選擇動作
        
        Args:
            observation: 環境觀察
            
        Returns:
            動作索引
        """
        with torch.no_grad():
            if self.model_type == 'QNetRNN':
                return self._select_action_rnn(observation)
            else:
                return self._select_action_qnet(observation)
    
    def _select_action_qnet(self, observation: np.ndarray) -> int:
        """QNet選擇動作"""
        obs_tensor = torch.tensor(observation, dtype=torch.float32, 
                                 device=self.device).unsqueeze(0)
        q_values = self.model(obs_tensor)
        return int(q_values.argmax(1).item())
    
    def _select_action_rnn(self, observation: np.ndarray) -> int:
        """QNetRNN選擇動作"""
        if self.hidden_state is None:
            self.hidden_state = self._init_hidden_state()
        
        obs_tensor = torch.tensor(observation, dtype=torch.float32,
                                 device=self.device).unsqueeze(0).unsqueeze(0)
        q_values, self.hidden_state = self.model(obs_tensor, self.hidden_state)
        return int(q_values.argmax(1).item())
    
    def update_reward(self, reward: float):
        """更新獎勵"""
        self.episode_reward += reward
        self.total_reward += reward
    
    def record_win(self):
        """記錄勝利"""
        self.wins += 1
    
    def record_loss(self):
        """記錄失敗"""
        self.losses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計數據"""
        total_games = self.wins + self.losses
        win_rate = self.wins / total_games if total_games > 0 else 0
        
        return {
            'name': self.name,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / total_games if total_games > 0 else 0
        }


class AgentPair:
    """代理對"""
    
    def __init__(self, agent_a: Agent, agent_b: Agent):
        """
        初始化代理對
        
        Args:
            agent_a: 上方代理
            agent_b: 下方代理
        """
        self.agent_a = agent_a
        self.agent_b = agent_b
    
    def reset_episode(self):
        """重置回合"""
        self.agent_a.reset_episode()
        self.agent_b.reset_episode()
    
    def select_actions(self, obs_a: np.ndarray, obs_b: np.ndarray) -> Tuple[int, int]:
        """
        選擇動作
        
        Args:
            obs_a: 代理A的觀察
            obs_b: 代理B的觀察
            
        Returns:
            (動作A, 動作B)
        """
        action_a = self.agent_a.select_action(obs_a)
        action_b = self.agent_b.select_action(obs_b)
        return action_a, action_b
    
    def update_rewards(self, reward_a: float, reward_b: float):
        """更新獎勵"""
        self.agent_a.update_reward(reward_a)
        self.agent_b.update_reward(reward_b)
    
    def record_result(self, score_a: int, score_b: int):
        """記錄比賽結果"""
        if score_a > score_b:
            self.agent_a.record_win()
            self.agent_b.record_loss()
        else:
            self.agent_a.record_loss()
            self.agent_b.record_win()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """獲取統計數據"""
        return {
            'agent_a': self.agent_a.get_stats(),
            'agent_b': self.agent_b.get_stats()
        }