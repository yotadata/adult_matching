# ユーザー埋め込み生成パイプライン

このドキュメントでは、ユーザーの行動履歴に基づいてユーザーの埋め込みベクトルを生成・更新するパイプラインについて説明します。

## 概要

ユーザーの埋め込みベクトルは、パーソナライズされたレコメンデーションを提供するために使用されます。このプロセスは、以下の主要なコンポーネントとステップで構成されます。

1.  **`embed-user` API (Supabase Edge Function)**
2.  **`update_user_features` RPC (PostgreSQL Function)**
3.  **`user_features` テーブル (PostgreSQL Table)**
4.  **`gen_user_embeddings.py` (バッチ処理)**
5.  **`user_embeddings` テーブル (PostgreSQL Table)**

## 1. `embed-user` API

### 目的

ユーザーの埋め込みベクトル生成の前段階となる「特徴量データ」の集計と保存を、任意のタイミングで特定のユーザーに対してトリガーします。このAPI自体は埋め込みベクトルを直接計算・保存するものではありません。

### エンドポイント

*   **URL:** `/functions/v1/embed-user`
*   **メソッド:** `POST`
*   **認証:** Supabase JWTによる認証が必須。

### 処理内容

1.  クライアントからの `POST` リクエストを受け付け、JWTでユーザーを認証します。
2.  認証された `user_id` を引数として、PostgreSQLのRPC `public.update_user_features` を呼び出します。
3.  RPCの実行結果に応じて、クライアントにステータスを返します。

### レスポンス例

```json
{
  "success": true,
  "user_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "message": "User features successfully updated."
}
```

## 2. `update_user_features` RPC

### 目的

特定のユーザーの行動履歴（いいね履歴）を集計し、そのユーザーの特徴量データを `user_features` テーブルに保存（upsert）します。

### 定義

`public.update_user_features(p_user_id uuid)`

### 処理内容

1.  引数 `p_user_id` で指定されたユーザーの `user_video_decisions` テーブルを参照します。
2.  ユーザーが「いいね (`like`)」した動画に関連するタグとパフォーマーをすべて集計します。
    *   集計結果は、タグIDごとのいいね回数、パフォーマーIDごとのいいね回数としてカウントされます。
3.  集計結果を `liked_tags` と `liked_performers` のJSONBオブジェクトとして構築します。
4.  構築したデータを `user_features` テーブルに `INSERT ON CONFLICT (user_id) DO UPDATE` (Upsert) します。

## 3. `user_features` テーブル

### 目的

ユーザーごとの集計済み特徴量データを格納します。このデータは、後続の埋め込みベクトル生成バッチ処理の入力となります。

### スキーマ

```sql
CREATE TABLE public.user_features (
    user_id uuid NOT NULL PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
    liked_tags jsonb NOT NULL DEFAULT '{}'::jsonb,
    liked_performers jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

### セキュリティ

*   Row Level Security (RLS) が有効化されており、ユーザー本人のみが自分の特徴量を閲覧・編集できます。
*   `service_role` には全アクセス権が付与されています。

## 4. `gen_user_embeddings.py` (バッチ処理)

### 目的

`user_features` テーブルに保存された特徴量データと最新のTwo-Towerモデルを使用して、ユーザーの埋め込みベクトルを生成し、`user_embeddings` テーブルを更新します。

### 処理内容

1.  `user_features` テーブルからユーザーごとの集計済み特徴量データを読み込みます。
2.  `train_two_tower.py` と同じロジック（`max_tag_features`, `max_performer_features` など）で、嗜好データからユーザーの特徴ベクトルを構築します。
3.  `publish_two_tower` で公開されている最新のTwo-Towerモデルの `user_encoder` を使用し、特徴ベクトルからユーザーの埋め込みベクトルを推論します。
4.  推論結果（`user_id`, `embedding`, `model_version`）を `user_embeddings` テーブルに `upsert` します。

### 実行タイミング

*   定期的なバッチ処理（例: cronジョブ）
*   手動でのトリガー

## 5. `user_embeddings` テーブル

### 目的

ユーザーの埋め込みベクトルを格納します。このベクトルは、動画レコメンデーションなどのパーソナライズ機能に使用されます。

### スキーマ

（既存のテーブル）

```sql
CREATE TABLE public.user_embeddings (
    user_id uuid NOT NULL PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
    embedding vector(256) NOT NULL,
    model_version text,
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

## パイプラインの連携

ユーザーが「いいね」などの行動を起こすと、クライアントは `embed-user` APIを呼び出します。これにより、`user_features` テーブルが更新されます。その後、`gen_user_embeddings.py` バッチ処理が実行されると、最新の `user_features` データに基づいてユーザーの埋め込みベクトルが再計算され、`user_embeddings` テーブルが更新されます。この更新された埋め込みベクトルは、`videos-feed` APIなどのレコメンデーション機能で利用されます。
