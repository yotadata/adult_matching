# 大規模データ収集実行ガイド

## 実行環境推奨仕様

### システム要件
- **OS**: Linux (Ubuntu 20.04+推奨) / WSL2
- **メモリ**: 4GB以上 (8GB推奨)
- **ストレージ**: 10GB以上の空き容量
- **ネットワーク**: 安定したインターネット接続

### Python環境
- **Python**: 3.8以上
- **uv**: 最新版
- **依存関係**: `uv sync` で自動インストール

## 実行オプション

### 1. フォアグラウンド実行（テスト・小規模）
```bash
# Step1: トップレビュワー抽出（5-10分）
make collect-top-reviewers

# Step2: レビュー収集（6-12時間）
make collect-large-reviews

# または一括実行
make collect-all
```

### 2. バックグラウンド実行（推奨・本番）
```bash
# バックグラウンド開始
make collect-large-bg

# 進捗確認
make collect-status

# ログ確認
make collect-logs

# 連続監視
make collect-watch
```

### 3. 中断・再開
```bash
# 中断時（Ctrl+C または kill）
# 進捗は自動保存される

# 再開
cd data_processing/scraping
./run_background.sh --resume
```

## データ保存方式

### ファイル構造
```
data_processing/raw_data/
├── top_reviewers.json              # Step1の結果
├── batch_reviews/                  # Step2の結果
│   ├── reviewer_61490.json         # レビュワー毎の個別ファイル  
│   ├── reviewer_185585.json
│   └── ...
├── scraping_progress.json          # 進捗状況
├── scraping.log                    # 実行ログ
└── scraper.pid                     # バックグラウンド実行時のPID
```

### データ追加方式
- **レビュワー単位**: 各レビュワーのレビューデータを個別JSONファイルに保存
- **追加保存**: 新しいレビュワーは追加、既存は上書き
- **中間保存**: 10ページごとに中間保存（データロス防止）
- **重複除外**: 既に完了済みのレビュワーは自動スキップ

### 推定データ量
- **レビュワー50人** × **平均2,000レビュー/人** = **約100,000レビュー**
- **1レビュー約1KB** = **総データ量約100MB**
- **JSON構造化後** = **約200-300MB**

## 実行パフォーマンス

### 実行時間（推定）
| フェーズ | 処理内容 | 推定時間 | 備考 |
|---------|---------|----------|------|
| Step1 | トップレビュワー抽出 | 5-10分 | 1ページのみ |
| Step2 | 50人×平均40ページ | 6-12時間 | レート制限込み |
| **合計** | **全体** | **6-12時間** | **連続実行の場合** |

### レート制限
- **ページ間**: 3秒間隔
- **レビュワー間**: 60秒間隔（5人ごと）
- **リトライ**: 失敗時3回まで自動リトライ
- **タイムアウト**: 30秒/リクエスト

## 監視・制御コマンド

### 進捗確認
```bash
make collect-status          # 現在の進捗を表示
make collect-logs           # 最新50行のログを表示
make collect-watch          # 30秒間隔で連続監視
```

### プロセス制御
```bash
# PIDファイルからプロセス確認
cat data_processing/raw_data/scraper.pid

# プロセス停止
kill $(cat data_processing/raw_data/scraper.pid)

# 強制停止
kill -9 $(cat data_processing/raw_data/scraper.pid)
```

### ログ監視
```bash
# リアルタイムログ監視
tail -f data_processing/raw_data/scraping.log

# エラーのみフィルタ
tail -f data_processing/raw_data/scraping.log | grep ERROR

# 進捗情報のみフィルタ  
tail -f data_processing/raw_data/scraping.log | grep "進捗:"
```

## トラブルシューティング

### よくある問題

#### 1. Cookie認証エラー
```bash
エラー: Cookie認証なしで継続
```
**解決策**: `data_processing/config/dmm_cookies.json` を設定

#### 2. 中断後の重複実行
```bash
エラー: スクレイパーは既に実行中です (PID: 12345)
```
**解決策**: 古いプロセスを停止後、PIDファイル削除
```bash
kill 12345
rm data_processing/raw_data/scraper.pid
```

#### 3. ディスク容量不足
**解決策**: 不要ファイル削除、または他のディスクに移動
```bash
df -h                        # 容量確認
make clean                   # 一時ファイル削除
```

#### 4. ネットワークエラー
```bash
ページ取得エラー (試行3): Connection timeout
```
**解決策**: 自動リトライ後、手動再開
```bash
./run_background.sh --resume
```

### ログレベル説明
- **INFO**: 正常な処理進捗
- **WARNING**: 警告（処理は継続）
- **ERROR**: エラー（該当レビュワーはスキップ）
- **DEBUG**: 詳細情報

### パフォーマンス調整

#### レート制限の調整
`robust_batch_scraper.py` の以下の値を変更:
```python
time.sleep(3)    # ページ間間隔（デフォルト3秒）
time.sleep(60)   # レビュワー間間隔（デフォルト60秒）
```

#### 並列処理（上級者向け）
現在はシーケンシャル処理ですが、必要に応じて並列処理に変更可能

## 成功後の次ステップ

### 1. データ統合
```bash
make data-clean                    # データクリーニング
make generate-pseudo-users         # 疑似ユーザー生成
```

### 2. ML学習
```bash
make data-embed                    # 埋め込みベクトル生成  
make train                         # Two-Towerモデル訓練
```

### 3. フルパイプライン
```bash
make train-full                    # 収集→学習の全工程
```

## 注意事項

### 法的・倫理的配慮
- ✅ robots.txt遵守
- ✅ レート制限実装済み
- ✅ 研究・開発目的での利用
- ✅ 個人情報の適切な取り扱い

### 技術的制約
- サーバー側の変更により動作しない可能性
- 大量アクセスによるIP制限の可能性
- Cookieの有効期限切れ

### データ品質
- 最低10文字未満のレビューは除外
- JSON形式での構造化済み
- 重複データの自動除去

---

**最終更新**: 2025年9月5日  
**作成者**: Claude Code  
**サポート**: 実行中の問題は進捗監視ツールまたはログで確認