# Pong Iterative Self-Play

本專案示範了如何建構一個「自我對戰」(Self-Play) 的 2D Pong 環境，並使用追加式的多世代訓練方式讓模型越來越強。
初始狀態下，兩個角色 (A, B) 共享相同的隨機權量；B 在每一世代中與 A 對戰並不斷學習，若 B 的勝率超過設定的關值，就將 B 視為更強的版本，並升級 A 與 B 同步為新權量，進入下一世代。

## 功能特色

1. **雙人對戰環境 (PongEnv2P)**  
   - 上板 (A) 與 下板 (B) 同時控制，各自具備 3 種離散動作：左移、停、右移。  
   - 簡化的球體邏輯（X 邊界反彈、Y 出界得分）。  
   - 回傳多智能體的觀測與獎勵 (可擴充成更多物理效果、旋轉、摩擦等)。

2. **多世代自對戰訓練**  
   - 初始 A, B 同一份隨機權量；B 不斷學習，A 不更新。  
   - 定期測試 B 對 A 的勝率，一旦超過關值 (win_threshold) => 升級。  
   - 無需固定 episode 數作為停止條件，讓模型自動決定「何時升級」。  
   - 可設定最大世代數 (max_generations) 以防無限學習。

3. **獨立 config.yaml**  
   - 在 `config.yaml` 調整學習率、追加世代數、勝率門檻、環境參數等。  
   - 可搭配其他檔案管理更多不同場景實驗。

4. **結果可視化**  
   - 產生 reward 曲線圖 (`training_iterative_rewards.png`)。  
   - 可進一步擴充紀錄勝率或對戰影片。

## 專案結構

```
pingpong-selfplay-ai/
├── envs/                      # 環境定義
│    ├── my_pong_env_2p.py    # 雙人Pong環境
│    └── physics.py           # 物理引擎相關
├── models/                    # 模型架構
│    ├── qnet.py              # QNet (DQN神經網路)
│    └── qnet_rnn.py          # QNetRNN (LSTM-based神經網路)
├── scripts/                   # 訓練腳本
│    ├── train_iterative.py   # QNet自對戰訓練
│    └── train_rnn_iterative.py # RNN自對戰訓練
├── tests/                     # 測試與評估工具
│    ├── test_viewer.py       # 2P Pong視覺化測試器
│    ├── test_round_robin.py  # 循環賽評估器
│    └── arena.py             # 智慧型競技場系統
├── checkpoints/               # QNet模型權重檔案
├── checkpoints_rnn/           # RNN模型權重檔案
├── results/                   # 訓練結果與分析
│    └── archive/             # 歷史結果存檔
├── assets/                    # 遊戲資源文件
├── config.yaml               # QNet訓練配置
├── config_rnn.yaml           # RNN訓練配置
├── arena_database.json       # 競技場對戰數據庫
├── README.md                 # 本說明文件
└── requirements.txt          # Python依賴套件
```

## 安裝與使用

1. **安裝依賴**  
   ```bash
   pip install -r requirements.txt
   ```

2. **執行訓練**  
   
   QNet模型訓練：
   ```bash
   python scripts/train_iterative.py
   ```
   
   RNN模型訓練：
   ```bash
   python scripts/train_rnn_iterative.py
   ```
   
   - 程式會讀取對應的配置檔案 (`config.yaml` 或 `config_rnn.yaml`)。  
   - 每個世代先執行一些對戰對局 (episodes_per_generation)，若 B 勝率 ≥ win_threshold => 升級。  
   - 產生檔案 `model_gen{N}.pth` 或 `rnn_pong_soul_{N}.pth`。

3. **模型測試與評估**
   
   視覺化測試（觀看兩個模型對戰）：
   ```bash
   python tests/test_viewer.py
   ```
   
   循環賽評估（多模型對戰統計）：
   ```bash
   python tests/test_round_robin.py
   ```
   
   競技場系統（持久化對戰數據庫）：
   ```bash
   python tests/arena.py
   ```

4. **查看訓練狀態**  
   - 於 console 可看到各世代 B 與 A 對戰的勝率、升級訊息。  
   - 訓練結果會儲存在 `results/` 目錄中。

## 參數調整

- 可在 `config.yaml` 裡面修改：  
  - `training > max_generations`: 世代上限  
  - `training > win_threshold`: 勝率門檻  
  - `training > eval_episodes`: 每次評估對局數  
  - `training > episodes_per_generation`: 每世代要對戰的場數  
  - `training > lr, gamma, batch_size, memory_size`: DQN 相關  
  - `env > render_size, paddle_width, paddle_speed, ball_speed, max_score`: Pong 2P 環境參數  

## 常見問題

1. **B 無法贏過 A**  
   - 可能需要調整 `episodes_per_generation` (更多學習機會)，或調整 hyperparams (lr, batch_size)。  

2. **想用更真實的物理**  
   - 可在 `my_pong_env_2p.py` 中加上旋轉、反彈公式、馬格努斯效應等邏輯。  

3. **如何紀錄 A, B 的歷史版本**  
   - 本示例只留下一個最新版本 (model_genX)；若要保留多世代對手，需自行增添存檔及管理策略池的機制。

## 貢獻

若你想擴充更多功能或提出建議，歡迎提交 Issue 或 Pull Request。

---

Enjoy your Self-Play Pong training!
