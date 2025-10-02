# Two-Tower Rebuild (v2)

本ディレクトリは Supabase の `videos` / `user_video_decisions` スキーマと互換を取りながら、Two-Tower レコメンドモデルを再構築するための Python パイプラインを収容しています。目的は次の 4 点です。

1. 疑似ユーザーデータ（`backend/data_processing/local_compatible_data`）や本番 DB から特徴量を抽出し、Two-Tower モデルを学習する。
2. 学習済み User Tower を ONNX 形式で出力し、Supabase Edge Functions（Deno/TypeScript + onnxruntime-web）から推論できるようにする。
3. 動画埋め込み（Item Tower 出力）を生成し、`video_embeddings` テーブルに一括アップサートできる成果物を出力する。
4. 推論に必要なメタ情報（特徴量スキーマ・正規化パラメータ・語彙辞書など）を JSON で出力し、CDN 経由で Edge Functions が取得できるようにする。

## ディレクトリ構成

```
backend/ml/two_tower_v2/
├── README.md
├── artifacts/                # エクスポート成果物配置先（例: user_tower.onnx, vocab_tag.json など）
├── config/                   # ハイパーパラメータや入出力設定
│   └── default.yaml
├── requirements.txt          # Python 依存関係
└── src/
    ├── data.py               # Supabase互換データのローディング・前処理
    ├── features.py           # 特徴量エンジニアリングとスキーマ生成
    ├── model.py              # PyTorch Two-Tower 実装
    ├── train.py              # 入口スクリプト（学習）
    ├── export.py             # ONNX および JSON 成果物の書き出し
    ├── generate_embeddings.py# Item Tower で全動画の埋め込みを生成
    └── utils.py              # 補助関数群
```

## 使い方

### 一括実行（推奨）

```
cd backend/ml/two_tower_v2
python run_local_pipeline.py
```

必要に応じて `--config` や入力データパスを上書きできます。エクスポートのみ/埋め込みのみをスキップしたい場合は `--skip-export` や `--skip-embeddings` オプションを指定してください。

### 手動実行（詳細）

1. 依存関係をインストール:
   ```bash
   cd backend/ml/two_tower_v2
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. 学習を実行:
   ```bash
   python src/train.py --config config/default.yaml \
     --profiles backend/data_processing/local_compatible_data/profiles.json \
     --videos backend/data_processing/local_compatible_data/videos_subset.json \
     --decisions backend/data_processing/local_compatible_data/user_video_decisions.json
   ```

3. 成果物を書き出し（上記コマンドで保存された `checkpoints/latest.pt` を利用）:
   ```bash
   python src/export.py --config config/default.yaml --checkpoint checkpoints/latest.pt --output-dir artifacts
   ```

4. 動画埋め込みを生成（Supabase にアップサートする CSV or Parquet を作成）:
   ```bash
   python src/generate_embeddings.py --config config/default.yaml --checkpoint checkpoints/latest.pt --videos backend/data_processing/local_compatible_data/videos_subset.json --output artifacts/video_embeddings.parquet
   ```

## 出力される成果物

- `user_tower.onnx` — Edge Functions から onnxruntime-web で読み込み可能な User Tower モデル
- `feature_schema.json` — 推論時に使用する特徴量の順序と型定義
- `normalizer.json` — 数値特徴量の平均・標準偏差
- `vocab_tag.json` / `vocab_actress.json` — 多値カテゴリ特徴量の語彙辞書
- `model_meta.json` — 署名、ハイパーパラメータ、学習日時を含むメタ情報
- `video_embeddings.parquet` — `video_id` とベクトル（`float32[embedding_dim]`）のテーブル

成果物一式は CI から CDN（例: Supabase Storage, CloudFront など）へアップロードし、Edge Functions で利用します。アップロード先のベース URL は `MODEL_ARTIFACT_BASE_URL` 環境変数として Edge Functions に設定してください。

## 今後の TODO

- 本番 DB から直接データを取得する SQL / Supabase API ラッパー
- A/B テストやメトリクス算出
- ハイパーパラメータチューニング用のスクリプト
