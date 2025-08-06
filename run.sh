#!/bin/bash
# Ping Pong Self-Play AI 啟動腳本

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ 虛擬環境不存在！${NC}"
    echo -e "${YELLOW}正在建立虛擬環境...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ 虛擬環境建立完成！${NC}"
else
    source venv/bin/activate
fi

# 主選單
while true; do
    clear
    echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     🏓 Ping Pong Self-Play AI 🏓        ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}請選擇要執行的功能：${NC}"
    echo ""
    echo "  1) 🎮 觀看模型對戰 (Visual Test)"
    echo "  2) 🏆 執行競技場排名賽 (Arena)"
    echo "  3) 🔄 快速循環賽 (Round Robin)"
    echo "  4) 🚀 訓練 Q-Network 模型"
    echo "  5) 🧠 訓練 RNN 模型"
    echo "  6) 📊 查看最新結果"
    echo "  7) 🔧 檢查環境設置"
    echo "  8) ❌ 退出"
    echo ""
    read -p "請輸入選項 (1-8): " choice

    case $choice in
        1)
            echo -e "${GREEN}啟動模型對戰視覺化...${NC}"
            python tests/test_viewer.py
            read -p "按 Enter 返回主選單..."
            ;;
        2)
            echo -e "${GREEN}開始競技場排名賽...${NC}"
            python tests/arena.py
            read -p "按 Enter 返回主選單..."
            ;;
        3)
            echo -e "${GREEN}執行快速循環賽...${NC}"
            python tests/test_round_robin.py
            read -p "按 Enter 返回主選單..."
            ;;
        4)
            echo -e "${GREEN}開始訓練 Q-Network...${NC}"
            python scripts/train_iterative.py
            read -p "按 Enter 返回主選單..."
            ;;
        5)
            echo -e "${GREEN}開始訓練 RNN 模型...${NC}"
            python scripts/train_rnn_iterative.py
            read -p "按 Enter 返回主選單..."
            ;;
        6)
            echo -e "${GREEN}最新結果：${NC}"
            echo ""
            if [ -f "results/summary.csv" ]; then
                echo "📊 最新對戰統計："
                cat results/summary.csv
            else
                echo "尚無結果檔案"
            fi
            echo ""
            if [ -f "results/h2h_heatmap.png" ]; then
                echo "🗺️  熱力圖已生成: results/h2h_heatmap.png"
            fi
            read -p "按 Enter 返回主選單..."
            ;;
        7)
            echo -e "${GREEN}環境檢查：${NC}"
            python -c "
import torch
import pygame
import gym
print('✅ Python 版本:', __import__('sys').version)
print('✅ PyTorch 版本:', torch.__version__)
print('✅ Pygame 版本:', pygame.version.ver)
print('✅ Gym 版本:', gym.__version__)
print('✅ MPS 可用 (Apple Silicon):', torch.backends.mps.is_available())
"
            read -p "按 Enter 返回主選單..."
            ;;
        8)
            echo -e "${YELLOW}再見！${NC}"
            deactivate 2>/dev/null
            exit 0
            ;;
        *)
            echo -e "${RED}無效選項！${NC}"
            sleep 1
            ;;
    esac
done