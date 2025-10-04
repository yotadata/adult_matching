# Two‑Tower モデル仕様と I/O（学習・推論）

本ドキュメントは、Two‑Tower（User/Item Embedding）モデルの設計方針、学習/推論時の入出力仕様、成果物（モデル/埋め込み）の取り扱い、および TypeScript / Python からの利用例をまとめます。

## 目的と全体像

- 目的: ユーザー嗜好に基づく動画レコメンド。類似度は内積/コサインで評価。
- 方式: Two‑Tower（ユーザー塔・アイテム塔）でそれぞれ埋め込みベクトルを学習し、`score = sim(E_u, E_i)` でランキング。
- 成果物:
  - 学習済みモデル（Storage）: `models/two_tower_latest.pkl`（任意でバージョニングファイル `two_tower_YYYYMMDDHHMM.pkl`）
  - 埋め込み（DB/pgvector）: `public.user_embeddings`, `public.video_embeddings`（vector(256) を想定）

## データ仕様（学習用）

- 入力データ: `interactions`（レビュー由来）
  - 生成元: `data/dmm_reviews.csv`（列: `product_url, reviewer_id, stars`）と `videos(product_url)` の突合
  - 最終スキーマ（Parquet/CSV）例: `user_id, item_id, label, weight, timestamp`
    - `user_id` = `reviewer_id`
    - `item_id` = `videos.id`（uuid）
    - `label` = 1/0（暗黙化: 例 `stars>=4` → 1、それ以外 0 へ負例サンプリング）
    - `weight` = 任意（例: `stars/5` を重み付けに使用可）
    - `timestamp` = 任意
- 負例サンプリング: `pos:neg = 1:K`（例 K=3）。K は学習スクリプト引数で制御。
- 分割: `train/valid = 8/2`（ユーザー層化推奨）

## モデル仕様

- エンコーダ
  - UserEncoder: `user_id -> R^D`（Embedding + MLP など）
  - ItemEncoder: `item_id -> R^D`（Embedding + MLP など）
- 出力次元 D: 256 を基本（計算都合により 128 などで学習→必要なら線形で 256 へ射影も可）
- 類似度: コサイン（DB は `vector_cosine_ops` を推奨）
- 損失: 対数双曲線/InfoNCE などの二値ロス or サンプリングベースのソフトマックス。初期は BCE + 負例サンプルで可。
- 最適化: Adam、`lr=1e-3` 目安、早期終了/学習率減衰オプション

## 成果物と配置

- Storage（Supabase）
  - `models/two_tower_latest.pkl`（pickle）
  - `models/two_tower_latest.json`（メタデータ: `version, trained_at, dim, loss, metrics`）
  - 版管理: `two_tower_YYYYMMDDHHMM.pkl` を保存し、`latest` を上書き
- DB（pgvector）
  - `public.video_embeddings(video_id uuid, embedding vector(256))`
  - `public.user_embeddings(user_id text, embedding vector(256))`
  - 初回のみ IVFFLAT インデックス作成（cosine）

```sql
create index if not exists idx_video_embeddings_cosine
  on public.video_embeddings using ivfflat (embedding vector_cosine_ops);
create index if not exists idx_user_embeddings_cosine
  on public.user_embeddings using ivfflat (embedding vector_cosine_ops);
```

## 推論 I/O 仕様

- 入力（いずれか）
  - `user_id`（既存ユーザー）→ `user_embeddings` から取得
  - 新規/未学習ユーザー: 簡易プロフィールや最近の行動から `E_u` を都度推定（当面は fallback: 人気混合）
  - 候補アイテム: `video_embeddings` 全体 or 事前絞り込み（新着・タグ一致など）
- 出力
  - `[{ video_id, score, rank }, ...]`（rank は score 降順）
  - UI では `videos` と join して `title/thumbnail/product_url` を返却

### TypeScript からの利用例（Supabase + pgvector）

```ts
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_ANON_KEY!);

async function recommendForUser(userId: string, k = 20) {
  // 1) user embedding を取得
  const { data: ue } = await supabase
    .from('user_embeddings')
    .select('embedding')
    .eq('user_id', userId)
    .single();
  if (!ue) return [];

  // 2) 類似検索（Edge Function / RPC でSQL発行する設計を推奨）
  // 例: SQL（サーバ側）: select video_id from video_embeddings order by embedding <-> $1 limit k;
}
```

Edge Function で RPC/SQL をまとめ、`embedding <-> $1` にユーザー埋め込みをバインドして返すのが安全です。

### Python からの利用例（psycopg + pgvector）

```python
import os
import numpy as np
import psycopg

conn = psycopg.connect(os.environ['DATABASE_URL'])
with conn.cursor() as cur:
    # user embedding を取得
    cur.execute("select embedding from public.user_embeddings where user_id=%s", (user_id,))
    ue = np.array(cur.fetchone()[0], dtype=float)

    # 類似検索
    cur.execute(
        "select video_id, embedding <-> %s as dist from public.video_embeddings order by dist asc limit %s",
        (ue.tolist(), k)
    )
    rows = cur.fetchall()
    video_ids = [r[0] for r in rows]
```

### モデルファイル（Storage）直接利用

- TypeScript（Deno/Node）からは pickle 直接読込は非推奨。モデルは基本 DB 埋め込み経由で利用。
- Python からは `requests` でダウンロード → `pickle` で `two_tower_latest.pkl` を読み込み、`encode_user/encode_item` を呼び出し可能（スクリプト側で公開インタフェースを提供）。

## 互換性と運用ポリシー

- `two_tower_latest.pkl` の I/F（エンコーダ関数名・戻り値次元）は後方互換を維持
- メタデータ JSON に `dim`/`version` を保持し、エッジ側が検証
- 失敗時フォールバック: 最新モデル取得に失敗→直近安定版 or popularity 混合に退避

---

本仕様に沿って、学習・推論スクリプトは安定した I/O を提供します。週次更新プロセスは `two_tower_weekly_update.md` を参照してください。
