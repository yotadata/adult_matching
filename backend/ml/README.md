# Adult Matching Backend ML Package

統一されたMLパイプラインパッケージです。Two-Tower推薦システムを中心とした機械学習コンポーネントを提供します。

## アーキテクチャ概要

```
backend/ml/
├── models/                    # トレーニング済みモデル
│   ├── rating_based_two_tower_768/  # 768次元標準モデル
│   ├── tfjs_exports/          # TensorFlow.js変換済みモデル
│   └── artifacts/             # モデル関連アーティファクト
├── training/                  # モデルトレーニング
│   ├── configs/               # トレーニング設定
│   ├── logs/                  # トレーニングログ
│   └── checkpoints/           # チェックポイント
├── preprocessing/             # データ前処理
│   ├── features/              # 特徴量エンジニアリング
│   └── embeddings/            # 埋め込み生成
├── inference/                 # 推論システム
│   ├── realtime/              # リアルタイム推論
│   └── batch/                 # バッチ推論
├── evaluation/                # モデル評価
│   ├── metrics/               # 評価指標
│   └── validation/            # バリデーション
├── deployment/                # デプロイメント
│   ├── supabase/              # Supabase統合
│   └── versioning/            # バージョン管理
├── monitoring/                # 監視システム
│   ├── performance/           # パフォーマンス監視
│   └── quality/               # 品質監視
└── utils/                     # ユーティリティ
```

## 標準768次元モデル

### モデル仕様
- **ユーザー埋め込み次元**: 768
- **アイテム埋め込み次元**: 768  
- **隠れ層**: [512, 256, 128]
- **正則化**: L2 (0.01)
- **ドロップアウト**: 0.2
- **学習率**: 0.001

### 使用方法

```python
from backend.ml import create_standard_trainer, create_standard_inference_engine

# トレーニング
trainer = create_standard_trainer()
trainer.train(user_data, item_data, interactions)

# 推論
engine = create_standard_inference_engine()
recommendations = engine.predict(user_id, candidate_items)
```

## モジュール詳細

### 1. Models (`models/`)
トレーニング済みモデルの保存・管理
- Keras形式の標準モデル
- TensorFlow.js変換済みモデル  
- モデルメタデータとバージョン情報

### 2. Training (`training/`)
モデルトレーニングパイプライン
- Two-Towerアーキテクチャの実装
- 分散トレーニング対応
- ハイパーパラメータ調整
- 早期停止・チェックポイント

### 3. Preprocessing (`preprocessing/`)
データ前処理とフィーチャーエンジニアリング
- ユーザー特徴量抽出
- アイテム特徴量抽出
- テキスト埋め込み生成
- データ正規化・スケーリング

### 4. Inference (`inference/`)
推論システム
- リアルタイム推論API
- バッチ推論処理
- 類似度計算
- レコメンデーション生成

### 5. Evaluation (`evaluation/`)
モデル評価とバリデーション
- オフライン評価指標
- A/Bテスト支援
- モデル比較
- パフォーマンス分析

### 6. Deployment (`deployment/`)
モデルデプロイメント
- Supabase Storage統合
- モデルバージョン管理
- ロールバック機能
- 自動デプロイメント

### 7. Monitoring (`monitoring/`)
運用監視
- リアルタイム品質監視
- パフォーマンス追跡
- アラート機能
- ダッシュボード

## 開発ガイドライン

### 1. モデルトレーニング
```python
# 標準設定でのトレーニング
from backend.ml import create_standard_trainer

trainer = create_standard_trainer(
    model_dir="./models/my_model_v1",
    config={
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.0005
    }
)

history = trainer.train(
    user_features=user_df,
    item_features=item_df, 
    interactions=interactions_df
)
```

### 2. モデル推論
```python
# リアルタイム推論
from backend.ml.inference.realtime_engine import RealtimeInferenceEngine

engine = RealtimeInferenceEngine(model_dir="./models/my_model_v1")
recommendations = engine.get_recommendations(
    user_id="user123",
    candidate_items=["item1", "item2", "item3"],
    top_k=10
)
```

### 3. モデルデプロイメント
```python
# Supabaseデプロイメント
from backend.ml.deployment.supabase_deployer import SupabaseDeployer

deployer = SupabaseDeployer()
deployment_info = deployer.deploy_model(
    model_dir="./models/my_model_v1",
    version="v1.2.0",
    description="768-dim Two-Tower with improved features"
)
```

## 設定ファイル

### トレーニング設定 (`training/configs/`)
```json
{
  "model": {
    "user_embedding_dim": 768,
    "item_embedding_dim": 768,
    "hidden_units": [512, 256, 128],
    "dropout_rate": 0.2,
    "l2_regularization": 0.01
  },
  "training": {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "patience": 10,
    "validation_split": 0.2
  }
}
```

### デプロイメント設定 (`deployment/configs/`)
```json
{
  "supabase": {
    "storage_bucket": "model-artifacts",
    "model_table": "model_versions",
    "auto_deployment": true
  },
  "versioning": {
    "strategy": "semantic",
    "rollback_enabled": true,
    "max_versions": 10
  }
}
```

## テストとQuality Assurance

### ユニットテスト
```bash
cd backend
python -m pytest ml/tests/unit/ -v
```

### 統合テスト
```bash
python -m pytest ml/tests/integration/ -v
```

### パフォーマンステスト
```bash  
python -m pytest ml/tests/performance/ -v
```

## ログとモニタリング

### ログレベル
- `DEBUG`: 詳細なデバッグ情報
- `INFO`: 一般的な処理情報
- `WARN`: 警告（処理は継続）
- `ERROR`: エラー（処理停止）

### ログファイル
- `ml/logs/training.log`: トレーニングログ
- `ml/logs/inference.log`: 推論ログ
- `ml/logs/deployment.log`: デプロイメントログ
- `ml/logs/monitoring.log`: 監視ログ

## 依存関係

### 主要パッケージ
- TensorFlow >= 2.16.0
- scikit-learn >= 1.3.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- psycopg2-binary >= 2.9.0

### 開発・テスト用
- pytest >= 7.4.0
- jupyter >= 1.0.0
- matplotlib >= 3.7.0

## パフォーマンス目標

### トレーニング
- **768次元モデル**: ~30分 (100エポック, 10万インタラクション)
- **メモリ使用量**: < 8GB
- **GPU使用率**: > 80%

### 推論
- **リアルタイム**: < 100ms (単一ユーザー推論)
- **バッチ**: > 1000 users/min
- **メモリ効率**: < 4GB (推論サーバー)

## トラブルシューティング

### 一般的な問題

1. **モデル読み込みエラー**
   - モデルファイルのパスを確認
   - バージョン互換性をチェック

2. **メモリ不足**
   - バッチサイズを削減
   - データのサンプリング

3. **推論速度低下**
   - モデルの軽量化を検討
   - キャッシュ機能の活用

詳細なドキュメントは各モジュールのREADMEを参照してください。