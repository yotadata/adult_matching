# プロジェクト構造

アダルト動画マッチングアプリケーションの整理されたプロジェクト構造です。

## 📁 ディレクトリ構成

```
adult_matching/
├── 📁 data_processing/          # データ処理関連
│   ├── 📁 scraping/            # ウェブスクレイピング
│   │   ├── dmm_review_scraper.py       # 基本スクレイパー
│   │   ├── dmm_advanced_scraper.py     # 高度スクレイパー
│   │   ├── cookie_dmm_scraper.py       # Cookie認証スクレイパー
│   │   └── __init__.py
│   ├── 📁 utils/               # データ処理ユーティリティ
│   │   ├── data_cleaner.py     # データクリーニング
│   │   └── __init__.py
│   ├── 📁 raw_data/            # 生データ格納
│   │   ├── dmm_reviews_*.json   # スクレイピング結果（JSON）
│   │   └── dmm_reviews_*.csv    # スクレイピング結果（CSV）
│   ├── 📁 processed_data/      # 前処理済みデータ
│   │   └── cleaned_reviews.json # クリーニング済みレビュー
│   ├── 📁 config/              # 設定ファイル
│   │   ├── dmm_cookies.json    # Cookie設定
│   │   └── Cookie取得手順.md    # Cookie取得ガイド
│   └── __init__.py
├── 📁 ml_pipeline/             # 機械学習パイプライン
│   ├── 📁 training/            # モデル訓練
│   │   ├── 📁 data/            # 訓練用データ
│   │   ├── train_two_tower_model.py    # Two-Towerモデル訓練
│   │   ├── run_batch_update.sh # バッチ更新スクリプト
│   │   └── __init__.py
│   ├── 📁 preprocessing/       # データ前処理
│   │   ├── batch_embedding_update.py  # 埋め込み更新
│   │   ├── review_to_embeddings.py    # レビュー→埋め込み変換
│   │   └── __init__.py
│   ├── 📁 models/              # 学習済みモデル格納
│   ├── 📁 evaluation/          # モデル評価
│   └── __init__.py
├── 📁 frontend/                # Next.jsフロントエンド
├── 📁 supabase/               # Supabaseバックエンド
├── 📁 scripts/                # その他スクリプト
├── 📁 docs/                   # ドキュメント
└── 📄 設定ファイル群
```

## 🔄 データフロー

### 1. データ収集フェーズ
```
1. data_processing/scraping/
   ├── cookie_dmm_scraper.py → DMM レビューサイトからデータ取得
   └── raw_data/ → 生データを JSON/CSV で保存
```

### 2. データ前処理フェーズ
```
2. data_processing/utils/
   ├── data_cleaner.py → テキストクリーニング・特徴量抽出
   └── processed_data/ → クリーニング済みデータを保存
```

### 3. 機械学習フェーズ
```
3. ml_pipeline/preprocessing/
   ├── review_to_embeddings.py → BERT埋め込みベクトル化
   └── training/data/ → 訓練用データセット作成

4. ml_pipeline/training/
   ├── train_two_tower_model.py → Two-Towerモデル訓練
   └── models/ → 学習済みモデル保存
```

## 🚀 使用方法

### データ収集の実行

```bash
# 1. Cookie設定（初回のみ）
# data_processing/config/Cookie取得手順.md を参照

# 2. レビューデータ収集
cd data_processing/scraping
uv run python cookie_dmm_scraper.py
```

### データ前処理の実行

```bash
# 1. データクリーニング
cd data_processing/utils
uv run python data_cleaner.py

# 2. 埋め込みベクトル変換
cd ../../ml_pipeline/preprocessing
uv run python review_to_embeddings.py
```

### モデル訓練の実行

```bash
# Two-Towerモデル訓練
cd ml_pipeline/training
uv run python train_two_tower_model.py
```

## 📋 各モジュールの役割

### data_processing/
- **目的**: 初期学習データの収集・前処理
- **機能**: 
  - ウェブスクレイピング（DMM、FANZAレビュー）
  - データクリーニング・正規化
  - 特徴量抽出

### ml_pipeline/
- **目的**: 機械学習モデルの訓練・評価・デプロイ
- **機能**:
  - テキスト埋め込みベクトル化
  - Two-Tower推薦モデル訓練
  - モデル評価・最適化

## ⚙️ 設定ファイル

### data_processing/config/
- **dmm_cookies.json**: DMM認証用Cookie設定
- **Cookie取得手順.md**: ブラウザからのCookie取得方法

## 🔧 依存関係

### Python環境
- 基本: `uv sync` でPython環境セットアップ
- 追加依存: `pandas`, `numpy`, `tensorflow`, `transformers`, `scikit-learn`

### データファイル形式
- **生データ**: JSON/CSV形式
- **前処理済み**: JSON形式（特徴量付き）
- **訓練データ**: NPZ形式（バイナリ、高速読み込み）

## 📊 データの流れ

```
DMM/FANZAレビューサイト
    ↓ (scraping)
生データ (raw_data/*.json)
    ↓ (cleaning)
クリーニング済みデータ (processed_data/*.json)
    ↓ (embedding)
訓練用データセット (training/data/*.npz)
    ↓ (training)
Two-Towerモデル (models/*.h5)
    ↓ (deployment)
Supabaseエッジファンクション
```

## 🛠️ 開発・運用

### 開発時
1. `data_processing/` でデータ収集・前処理
2. `ml_pipeline/` でモデル開発・訓練
3. `frontend/` でUI開発
4. `supabase/` でバックエンドAPI開発

### 本番運用時
1. 定期的なデータ収集（スケジューラー）
2. モデル再訓練（バッチ処理）
3. モデルデプロイ（Supabaseエッジファンクション更新）

この構造により、データ処理とML機能が明確に分離され、保守性と拡張性が向上しています。