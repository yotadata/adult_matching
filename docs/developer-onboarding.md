# 開発者オンボーディングガイド

## 📋 概要

Adult Matchingバックエンドシステムの新規開発者向けオンボーディングガイドです。このガイドでは、開発環境のセットアップから実際の開発作業まで、段階的に説明します。

## 🎯 学習目標

このガイドを完了すると、以下のことができるようになります：

1. **環境構築**: 開発環境の完全セットアップ
2. **アーキテクチャ理解**: システム全体構成の把握
3. **開発フロー**: 標準開発プロセスの実践
4. **テスト実行**: 包括的テスト実行
5. **デプロイメント**: 安全なデプロイメント実行
6. **トラブルシューティング**: 問題解決手法

## 🚀 セットアップ手順

### ステップ1: 前提条件確認

**必要なソフトウェア:**
```bash
# Node.js (v18+)
node --version  # v18.0.0以上

# Python (3.12+)
python3 --version  # 3.12.0以上

# Supabase CLI
supabase --version  # 1.100.0以上

# Docker & Docker Compose
docker --version
docker-compose --version

# Git
git --version
```

**推奨ツール:**
- **IDE**: VSCode + 推奨拡張機能
- **ターミナル**: iTerm2 (macOS) / Windows Terminal
- **データベースクライアント**: pgAdmin / DBeaver

### ステップ2: リポジトリクローンと初期設定

```bash
# リポジトリクローン
git clone https://github.com/your-org/adult-matching.git
cd adult-matching

# Pythonパッケージマネージャセットアップ
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# Python環境構築
uv sync

# Node.js依存関係インストール (フロントエンド用)
cd frontend
npm install
cd ..
```

### ステップ3: 環境変数設定

```bash
# メイン環境変数ファイル作成
cp .env.example .env

# Supabase環境変数ファイル作成
cp supabase/.env.example supabase/.env
```

**.env設定例:**
```bash
# Supabase設定
SUPABASE_URL=http://localhost:54321
SUPABASE_ANON_KEY=your_local_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_local_service_role_key

# DMM API設定
DMM_API_ID=your_dmm_api_id
DMM_AFFILIATE_ID=your_dmm_affiliate_id

# 監視設定
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO

# 開発設定
ENVIRONMENT=development
DEBUG=true
```

### ステップ4: データベースセットアップ

```bash
# Supabaseローカル環境起動
supabase start

# データベース移行実行
supabase db reset

# 初期データ投入（オプション）
python backend/data/sync/dmm/real_dmm_sync.py --limit 100
```

### ステップ5: Edge Functions デプロイ

```bash
# 全Edge Functionsデプロイ
supabase functions deploy

# または個別デプロイ
supabase functions deploy enhanced_two_tower_recommendations
supabase functions deploy user-management-likes
supabase functions deploy content-feed
```

### ステップ6: 開発サーバー起動

```bash
# バックエンド監視システム起動
python -m backend.monitoring &

# フロントエンド開発サーバー起動
cd frontend
npm run dev &
cd ..

# MLトレーニングテスト実行（オプション）
python -m backend.ml.training.test_basic_training
```

## 🏗️ アーキテクチャ理解

### システム全体像

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │ ←→ │ Edge Functions  │ ←→ │ Backend Services│
│   (Next.js)     │    │   (Supabase)    │    │  (Python/ML)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↕                       ↕                       ↕
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │   Monitoring    │    │  Infrastructure │
│ (PostgreSQL)    │    │ (Prometheus)    │    │ (Docker/CI/CD)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 主要コンポーネント

**1. Edge Functions (API層)**
- `supabase/functions/` - サーバーレス API エンドポイント
- 認証、ビジネスロジック、データ取得を担当
- TypeScript実装、自動スケーリング

**2. Backend Services (処理層)**
- `backend/ml/` - 機械学習パイプライン
- `backend/data/` - データ管理・同期
- `backend/monitoring/` - 監視・ログ
- `backend/optimization/` - パフォーマンス最適化

**3. Database (データ層)**
- PostgreSQL + pgvector
- 768次元ベクター埋め込み
- Row Level Security (RLS)

### データフロー例: 推薦生成

```
1. User Request → Frontend
2. Frontend → Edge Function (enhanced_two_tower_recommendations)
3. Edge Function → Backend ML Service
4. ML Service → Database (ベクター検索)
5. Database → ML Service (候補動画)
6. ML Service → Edge Function (ランキング)
7. Edge Function → Frontend (推薦結果)
8. 監視システム ← 全ステップでメトリクス収集
```

## 🛠️ 開発ワークフロー

### 標準開発プロセス

**1. 機能要件確認**
```bash
# 既存タスクファイル確認
cat .spec-workflow/specs/backend-refactoring/tasks.md

# 関連文書確認
ls docs/specifications/
```

**2. ブランチ作成**
```bash
# 機能ブランチ作成
git checkout -b feature/new-recommendation-algorithm

# または修正ブランチ
git checkout -b fix/user-embedding-update-bug
```

**3. 開発実行**
```bash
# 関連コンポーネント特定
find . -name "*.py" -o -name "*.ts" | grep -E "(recommendation|embedding)"

# テスト駆動開発
pytest backend/tests/unit/ml/test_new_algorithm.py -v

# 実装
# コード実装...

# 動作確認
python -m backend.ml.test_new_algorithm
```

**4. テスト実行**
```bash
# 単体テスト
pytest backend/tests/unit/ -v

# 統合テスト
pytest backend/tests/integration/ -v

# E2Eテスト
pytest backend/tests/e2e/ -v

# カバレッジ確認
pytest --cov=backend --cov-report=html
```

**5. コード品質チェック**
```bash
# Python linting
ruff check backend/
black backend/

# TypeScript linting
cd supabase/functions
npm run lint
cd ../..

# 型チェック
mypy backend/
npx tsc --noEmit
```

**6. パフォーマンステスト**
```bash
# パフォーマンス検証
python backend/optimization/performance_verifier.py

# 負荷テスト
python backend/tests/integration/performance/load_test.py
```

### コミット・プッシュ

```bash
# 変更内容確認
git status
git diff

# ステージング
git add .

# コミット（規約に従う）
git commit -m "feat: Add new recommendation algorithm

- Implement collaborative filtering enhancement
- Add performance optimization for large datasets
- Include comprehensive unit tests
- Update documentation

Fixes #123"

# プッシュ
git push origin feature/new-recommendation-algorithm
```

### プルリクエスト

**PR作成時チェックリスト:**
- [ ] 全テストが通過
- [ ] コードカバレッジ ≥95%
- [ ] パフォーマンステスト通過
- [ ] セキュリティチェック通過
- [ ] 文書更新完了
- [ ] 後方互換性確認

## 🧪 テスト戦略

### テスト分類と実行方法

**1. 単体テスト**
```bash
# ML コンポーネント
pytest backend/tests/unit/ml/ -v

# データ処理
pytest backend/tests/unit/data/ -v

# Edge Functions
pytest backend/tests/unit/functions/ -v
```

**2. 統合テスト**
```bash
# MLパイプライン統合
pytest backend/tests/integration/ml/ -v

# データパイプライン統合
pytest backend/tests/integration/data/ -v

# API統合
pytest backend/tests/integration/content/ -v
```

**3. E2Eテスト**
```bash
# ユーザーワークフロー
pytest backend/tests/e2e/user_workflows/ -v

# パフォーマンステスト
pytest backend/tests/e2e/performance/ -v
```

### テストデータ管理

```bash
# テストデータ生成
python backend/tests/fixtures/generate_test_data.py

# テストデータベースリセット
supabase db reset --db-url postgresql://postgres:postgres@localhost:54322/postgres
```

## 🚀 デプロイメント

### 開発環境デプロイ

```bash
# Edge Functions更新
supabase functions deploy enhanced_two_tower_recommendations

# データベース移行
supabase db push

# バックエンドサービス再起動
python -m backend.monitoring.restart_services
```

### ステージング環境デプロイ

```bash
# ステージング環境デプロイ
./backend/scripts/deployment/deploy.sh staging

# デプロイ後検証
./backend/scripts/deployment/verify_deployment.sh staging
```

### 本番環境デプロイ

```bash
# 本番環境デプロイ（要承認）
./backend/scripts/deployment/deploy.sh production

# 監視ダッシュボード確認
# http://localhost:3000/grafana

# ロールバック（必要時）
./backend/scripts/deployment/rollback.sh
```

## 🔧 開発ツールと設定

### VSCode推奨拡張機能

**.vscode/extensions.json:**
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "bradlc.vscode-tailwindcss",
    "supabase.supabase",
    "ms-vscode.vscode-typescript-next",
    "esbenp.prettier-vscode",
    "ms-vscode.test-adapter-converter"
  ]
}
```

### VSCode設定

**.vscode/settings.json:**
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "typescript.preferences.importModuleSpecifier": "relative",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll.ruff": true
  }
}
```

### デバッグ設定

**.vscode/launch.json:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug ML Training",
      "type": "python",
      "request": "launch",
      "module": "backend.ml.training.train_two_tower",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Debug Edge Function",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/supabase/functions/enhanced_two_tower_recommendations/index.ts",
      "outFiles": ["${workspaceFolder}/supabase/functions/**/*.js"]
    }
  ]
}
```

## 📊 監視とログ

### 開発時監視

```bash
# システム監視起動
python -m backend.monitoring.system_monitor

# 統合監視起動
python -m backend.monitoring.integration_monitor

# ログ監視
tail -f backend/logs/application.log
```

### メトリクス確認

```bash
# Prometheusメトリクス確認
curl http://localhost:8000/metrics

# システム状態確認
python -c "
from backend.monitoring import create_system_monitor
import asyncio
monitor = create_system_monitor()
metrics = asyncio.run(monitor.collect_system_metrics())
print(f'CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%')
"
```

### ログレベル設定

```python
# backend/monitoring/logger_config.py
import logging

DEVELOPMENT_CONFIG = {
    'level': logging.DEBUG,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file']
}

PRODUCTION_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'handlers': ['file', 'syslog']
}
```

## 🐛 トラブルシューティング

### よくある問題と解決策

**1. Supabase接続エラー**
```bash
# 問題: Supabaseサービスが起動しない
# 解決:
supabase stop
supabase start

# ポート競合確認
lsof -i :54321
```

**2. Edge Function デプロイエラー**
```bash
# 問題: Function deploymentが失敗する
# 解決:
supabase functions delete function_name
supabase functions deploy function_name

# ログ確認
supabase functions logs function_name
```

**3. ML トレーニングメモリエラー**
```bash
# 問題: Out of Memory during training
# 解決:
# バッチサイズ削減
export ML_BATCH_SIZE=256

# メモリ最適化設定
export ML_MEMORY_OPTIMIZATION=true

python -c "
from backend.ml.training import create_ml_training_optimizer
optimizer = create_ml_training_optimizer()
optimizer.optimize_memory_usage()
"
```

**4. パフォーマンス問題**
```bash
# 診断実行
python backend/optimization/performance_verifier.py

# プロファイリング
python -m cProfile -o profile.stats backend/ml/training/train_two_tower.py

# 結果確認
python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative').print_stats(10)
"
```

**5. データベース問題**
```bash
# 接続数確認
psql postgresql://postgres:postgres@localhost:54322/postgres -c "
SELECT count(*) FROM pg_stat_activity;
"

# インデックス再構築
python -c "
from backend.data.maintenance import rebuild_indexes
rebuild_indexes()
"

# ベクター検索最適化
python -c "
from backend.optimization import DatabaseOptimizer
optimizer = DatabaseOptimizer()
optimizer.optimize_vector_search()
"
```

### デバッグ手法

**1. ログベースデバッグ**
```python
import logging

# 詳細ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 重要ポイントでログ出力
logger.debug(f"Processing user {user_id} with {len(videos)} videos")
logger.info(f"Recommendation generated in {duration:.2f}ms")
```

**2. パフォーマンスプロファイリング**
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {(end - start) * 1000:.2f}ms")

# 使用例
with timer("Vector Search"):
    results = search_similar_videos(user_embedding)
```

**3. メモリ使用量監視**
```python
import psutil
import tracemalloc

# メモリ追跡開始
tracemalloc.start()

# 処理実行
process_large_dataset()

# メモリ使用量確認
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## 🎓 実践演習

### 演習1: 新しい推薦アルゴリズム実装

**目標**: ジャンルベース推薦アルゴリズムを実装

```python
# backend/ml/algorithms/genre_based_recommender.py
class GenreBasedRecommender:
    def __init__(self):
        self.genre_weights = {}

    def train(self, user_likes_data):
        # 実装してください
        pass

    def recommend(self, user_id, num_recommendations=10):
        # 実装してください
        pass

# テスト実装
# backend/tests/unit/ml/test_genre_based_recommender.py
```

**期待される学習成果**:
- MLコンポーネント設計理解
- テスト駆動開発実践
- パフォーマンス考慮実装

### 演習2: 新しいEdge Function作成

**目標**: 動画統計情報API作成

```typescript
// supabase/functions/video-stats/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

serve(async (req) => {
  // 実装してください
  return new Response(
    JSON.stringify({ success: true, data: stats }),
    { headers: { "Content-Type": "application/json" } }
  )
})
```

**期待される学習成果**:
- Edge Function アーキテクチャ理解
- Supabase統合理解
- API設計ベストプラクティス

### 演習3: 監視メトリクス追加

**目標**: カスタムメトリクス実装

```python
# backend/monitoring/custom_metrics.py
from prometheus_client import Counter, Histogram

video_view_counter = Counter(
    'adult_matching_video_views_total',
    'Total video views',
    ['genre', 'maker']
)

recommendation_latency = Histogram(
    'adult_matching_recommendation_duration_seconds',
    'Time spent generating recommendations'
)

# 使用方法実装
```

**期待される学習成果**:
- 監視システム設計理解
- Prometheusメトリクス理解
- 運用観点の考慮

## 📚 学習リソース

### 必読ドキュメント

1. **[backend/README.md](../backend/README.md)** - システム概要
2. **[docs/backend-architecture.md](backend-architecture.md)** - アーキテクチャ詳細
3. **[Supabase Documentation](https://supabase.com/docs)** - Supabase公式文書
4. **[PostgreSQL + pgvector](https://github.com/pgvector/pgvector)** - ベクター検索

### 技術スタック学習

**Python/ML:**
- TensorFlow 2.16+ 公式チュートリアル
- scikit-learn User Guide
- pandas Documentation

**TypeScript/Edge Functions:**
- TypeScript Handbook
- Deno Manual
- Supabase Edge Functions Guide

**Database:**
- PostgreSQL Official Tutorial
- pgvector README
- SQL Performance Tuning

### 実践プロジェクト

**レベル1: 基本操作**
- Edge Function 1つ作成・デプロイ
- 単体テスト作成・実行
- 基本的な監視メトリクス確認

**レベル2: 統合開発**
- 新機能エンドツーエンド実装
- 統合テスト作成
- パフォーマンステスト実行

**レベル3: アーキテクチャ拡張**
- 新しいMLアルゴリズム実装
- システム最適化実施
- 監視・アラート設定

## 🤝 コミュニティ

### チーム連携

**コミュニケーション:**
- Slack: `#backend-dev` チャンネル
- 週次開発者ミーティング: 毎週水曜 10:00
- コードレビュー: GitHub PR必須

**ナレッジ共有:**
- 技術ブログ記事執筆
- 社内技術勉強会プレゼン
- ペアプログラミング実施

### コントリビューション

**コード品質基準:**
- テストカバレッジ ≥95%
- 型安全性厳守
- パフォーマンス要件達成
- セキュリティベストプラクティス
- 包括的文書化

**レビュープロセス:**
1. 自己レビュー実施
2. ペアレビュー依頼
3. シニア開発者レビュー
4. CI/CDパイプライン通過
5. マージ・デプロイ

---

## 📞 サポート

**技術的質問:**
- Slack: `#backend-help`
- Email: backend-team@company.com

**緊急時対応:**
- On-call Engineer: +81-XX-XXXX-XXXX
- エスカレーション: CTO直通

**学習サポート:**
- メンター制度利用可能
- 外部研修制度あり
- 技術書購入支援

---

**ガイドバージョン**: 1.0
**最終更新**: 2025年9月16日
**作成者**: Claude Code Assistant
**次回更新予定**: 2025年12月16日

このガイドを完了すると、Adult Matchingバックエンドシステムの効率的な開発が可能になります。不明な点があれば、いつでもチームにご相談ください！