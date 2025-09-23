# Backend Architecture Design Document
# バックエンド統合アーキテクチャ設計書

## 概要

Adult Matching バックエンドシステムの統合アーキテクチャです。機械学習、データ処理、テスト、デプロイメントを統一的に管理する設計になっています。

## 🏗️ 最終的なディレクトリ構造

```
backend/
├── __init__.py                          # バックエンドパッケージ初期化
├── README.md                            # バックエンド全体のドキュメント
├── ARCHITECTURE.md                      # 本設計書
│
├── ml/                                  # 🤖 機械学習システム
│   ├── __init__.py
│   ├── README.md
│   ├── scripts/                         # MLスクリプト
│   │   ├── training/                    # 訓練スクリプト
│   │   │   ├── train_768_dim_two_tower.py
│   │   │   ├── train_production_two_tower.py
│   │   │   ├── train_standard_model.py
│   │   │   ├── train_two_tower_comprehensive.py
│   │   │   └── train_two_tower_model.py
│   │   ├── testing/                     # テストスクリプト
│   │   │   ├── simple_two_tower_test.py
│   │   │   ├── test_768_pgvector_integration.py
│   │   │   ├── test_training_components.py
│   │   │   └── verify_768_model.py
│   │   ├── deployment/                  # デプロイメントスクリプト
│   │   │   └── model_deployment.py
│   │   └── standardize_models_768.py
│   ├── training/                        # 訓練コンポーネント
│   │   ├── __init__.py
│   │   ├── trainers/
│   │   │   └── unified_two_tower_trainer.py
│   │   └── configs/
│   │       └── standard_768_config.json
│   ├── preprocessing/                   # 前処理コンポーネント
│   │   ├── __init__.py
│   │   ├── features/                    # 特徴処理
│   │   │   ├── feature_processor.py
│   │   │   ├── user_feature_processor.py
│   │   │   └── item_feature_processor.py
│   │   ├── embeddings/                  # 埋め込み管理
│   │   │   ├── __init__.py
│   │   │   └── embedding_manager.py
│   │   └── utils/                       # 前処理ユーティリティ
│   │       ├── feature_scaler.py
│   │       └── data_validator.py
│   ├── deployment/                      # デプロイメント
│   │   ├── __init__.py
│   │   └── tensorflowjs_converter.py
│   └── utils/                           # MLユーティリティ
│       ├── logger.py
│       ├── config_loader.py
│       └── gpu_manager.py
│
├── data/                                # 📊 データ管理システム
│   ├── __init__.py                      # 統合データ管理パッケージ
│   ├── ingestion/                       # データ取り込み
│   │   ├── __init__.py
│   │   └── data_ingestion_manager.py
│   ├── processing/                      # データ処理
│   │   ├── __init__.py
│   │   └── unified_data_processor.py
│   ├── validation/                      # データ検証
│   │   ├── __init__.py
│   │   └── data_validator.py
│   ├── export/                          # データエクスポート
│   │   ├── __init__.py
│   │   └── export_manager.py
│   ├── schemas/                         # スキーマ管理
│   │   ├── __init__.py
│   │   └── schema_manager.py
│   ├── pipelines/                       # パイプライン管理
│   │   ├── __init__.py
│   │   └── pipeline_manager.py
│   ├── utils/                           # データユーティリティ
│   └── storage/                         # 📁 データストレージ
│       ├── raw/                         # 生データ
│       ├── processed/                   # 処理済みデータ
│       ├── validated/                   # 検証済みデータ
│       ├── exports/                     # エクスポートデータ
│       ├── cache/                       # キャッシュ
│       ├── models/                      # 🎯 訓練済みモデル保存
│       │   ├── two_tower_pattern1/      # Two-Towerモデル
│       │   └── comprehensive_two_tower_pattern1/
│       ├── logs/                        # ログ
│       ├── schemas/                     # スキーマファイル
│       └── temp/                        # 一時ファイル
│
├── tests/                               # 🧪 テストシステム
│   ├── __init__.py
│   ├── test_framework.py                # 統合テストフレームワーク
│   ├── logs/                            # テストログ
│   └── test_report.json                 # テストレポート
│
├── scripts/                             # ⚙️ 管理スクリプト
│   ├── unified_script_runner.py         # 統合スクリプト実行システム
│   ├── script_migration_manager.py      # スクリプト移行管理
│   ├── migration_report.json            # 移行レポート
│   ├── deployment/                      # 🚀 インフラデプロイメント
│   │   ├── docker/                      # Docker設定
│   │   │   ├── Dockerfile.backend
│   │   │   └── docker-compose.yml
│   │   ├── kubernetes/                  # Kubernetes設定
│   │   │   └── backend-deployment.yaml
│   │   ├── monitoring/                  # 監視設定
│   │   │   ├── prometheus.yml
│   │   │   └── alert_rules.yml
│   │   └── optimization/                # 最適化設定
│   ├── organized/                       # 整理済みスクリプト
│   │   ├── ml-training/
│   │   ├── data-processing/
│   │   ├── api-sync/
│   │   ├── analysis/
│   │   ├── deployment/
│   │   ├── maintenance/
│   │   └── legacy/
│   ├── batch-processing/
│   └── data-migration/
│
└── utils/                               # 🔧 共通ユーティリティ
    ├── __init__.py
    ├── logger.py                        # ログユーティリティ
    ├── config.py                        # 設定管理
    ├── database.py                      # データベースユーティリティ
    └── helpers.py                       # ヘルパー関数
```

## 🎯 主要な設計特徴

### 1. **統合アーキテクチャ**
- **単一責任の原則**: 各ディレクトリが明確な責任を持つ
- **階層化設計**: 機能別の明確な階層構造
- **統合管理**: 統一されたインターフェースでの管理

### 2. **スケーラビリティ**
- **モジュラー設計**: 独立したコンポーネント
- **拡張性**: 新機能追加時の構造保持
- **再利用性**: 共通コンポーネントの活用

### 3. **運用性**
- **統合スクリプト実行**: `unified_script_runner.py`
- **包括的テスト**: 自動化されたテストフレームワーク
- **監視・ログ**: 本格的な運用監視システム

## 🔄 データフロー

```
Raw Data → Ingestion → Processing → Validation → Storage
    ↓
ML Training → Model Storage → Deployment → Inference
    ↓
Monitoring ← Logs ← Performance Metrics
```

## 🚀 統合システムの使用方法

### 1. **スクリプト実行**
```bash
# 全スクリプト一覧
python backend/scripts/unified_script_runner.py list

# MLスクリプト実行
python backend/scripts/unified_script_runner.py run train_768_two_tower

# DMM同期実行
python backend/scripts/unified_script_runner.py run dmm_sync
```

### 2. **データ管理**
```python
from backend.data import create_data_manager

# 統合データ管理システム
managers = create_data_manager()
ingestion = managers["ingestion"]
processing = managers["processing"]
validation = managers["validation"]
```

### 3. **ML訓練**
```python
from backend.ml.training.trainers.unified_two_tower_trainer import UnifiedTwoTowerTrainer

trainer = UnifiedTwoTowerTrainer()
model = trainer.train(config_path="backend/ml/training/configs/standard_768_config.json")
```

### 4. **テスト実行**
```bash
python backend/tests/test_framework.py
```

## 🗂️ 削除済みディレクトリ

以下の旧ディレクトリは統合により削除されました：

- ❌ `backend/data-processing/` → `backend/data/` に統合
- ❌ `backend/edge_functions/` → `supabase/functions/` に移行
- ❌ `backend/ml-pipeline/` → `backend/ml/` に統合
- ❌ `backend/models/` → `backend/data/storage/models/` に移行
- ❌ `backend/deployment/` → `backend/scripts/deployment/` に移行

## 📋 統合の利点

### 1. **一貫性**
- 統一されたディレクトリ構造
- 標準化されたインターフェース
- 共通の設定・ログ管理

### 2. **効率性**
- 重複コードの排除
- 共通コンポーネントの再利用
- 統合された実行環境

### 3. **保守性**
- 明確な責任分離
- 簡単な機能追加・変更
- 包括的なテスト・監視

## 🔧 開発・運用ガイドライン

### 1. **新機能追加**
- 適切なディレクトリに配置
- 統合スクリプトランナーに登録
- テストケースの追加

### 2. **デプロイメント**
- `backend/scripts/deployment/` の設定を使用
- 段階的デプロイメント
- 監視システムでの確認

### 3. **データ処理**
- `backend/data/` パッケージの統一インターフェース使用
- スキーマ定義・検証の実施
- パイプライン管理による自動化

この統合アーキテクチャにより、Adult Matching バックエンドシステムは効率的で保守性の高いシステムとなります。