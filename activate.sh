#!/bin/bash
# 快速啟動虛擬環境腳本

if [ ! -d "venv" ]; then
    echo "建立虛擬環境..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ 虛擬環境建立完成！"
else
    source venv/bin/activate
    echo "✅ 虛擬環境已啟動！"
fi

echo ""
echo "可用的快速指令："
echo "  python tests/test_viewer.py     # 觀看模型對戰"
echo "  python tests/arena.py           # 執行競技場賽"
echo "  python tests/test_round_robin.py # 快速循環賽"
echo "  ./run.sh                        # 互動式選單"
echo ""
echo "輸入 'deactivate' 來退出虛擬環境"

# 保持在虛擬環境中
exec $SHELL