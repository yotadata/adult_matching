# videos-feed エッジ関数仕様

このドキュメントは Edge Function `videos-feed` が返すレコメンドリストの構成と、バックエンドで用いている RPC の仕様を整理したものです。サービス運用時のチューニングや A/B テストのポイントも併記しています。

## 目的

- Two-Tower で生成した埋め込みを用いてユーザー個別のレコメンドを提示する。
- 完全なランキング結果だけでなく、人気枠や探索枠を混在させ、偏りを抑えながら多様な作品を提示する。
- レスポンスに `score` と `model_version` を含め、動作確認・バージョン追跡を可能にする。

## 全体フロー

```
リクエスト (JWT あり/なし, limit, popularity_days)
        │
        ├─ JWT あり → Supabase Auth getUser()
        │      │
        │      └─ OK → exploitation (推薦) を取得
        │
        ├─ popularity RPC → 最近の人気作品候補を取得
        │
        ├─ exploration RPC → ランダムな候補を取得
        │
        └─ exploitation/popularity/exploration を比率に応じてマージ → 返却
```

### パラメータ

| 項目 | 既定値 | 備考 |
| ---- | ------ | ---- |
| `limit` | 20 | `MAX_LIMIT=50` まで拡張可能 |
| `popularity_days` | 7 | 人気枠の Lookback 日数 |
| `exploitation_ratio` | 0.6 | 推薦枠の目標比率 |
| `popularity_ratio` | 0.2 | 人気枠の目標比率 |
| `exploration` | 残差 | 全体が `limit` を満たすよう調整 |

### レスポンス構造（例）

```jsonc
[
  {
    "id": "6c14e381-396d-4839-a6ba-d80a347b1141",
    "title": "...",
    "score": 0.7039,
    "model_version": "20251031_183831",
    "source": "exploitation",
    "params": {
      "requested_limit": 20,
      "exploitation_ratio": 0.6,
      "popularity_ratio": 0.2,
      "exploration_ratio": 0.2,
      "popularity_lookback_days": 7,
      "exploitation_returned": 12,
      "popularity_returned": 4,
      "exploration_returned": 4
    }
  },
  {
    "id": "4f8e77b2-a4e3-4d10-ac75-84a94b1cf2f9",
    "score": 127,
    "model_version": null,
    "source": "popularity",
    "params": {
      "requested_limit": 20,
      "exploitation_ratio": 0.6,
      "popularity_ratio": 0.2,
      "exploration_ratio": 0.2,
      "popularity_lookback_days": 7,
      "exploitation_returned": 12,
      "popularity_returned": 4,
      "exploration_returned": 4
    }
  },
  {
    "id": "9af83f62-07a8-49b4-bf4c-8f4c0429b2f2",
    "score": null,
    "model_version": null,
    "source": "exploration",
    "params": {
      "requested_limit": 20,
      "exploitation_ratio": 0.6,
      "popularity_ratio": 0.2,
      "exploration_ratio": 0.2,
      "popularity_lookback_days": 7,
      "exploitation_returned": 12,
      "popularity_returned": 4,
      "exploration_returned": 4
    }
  }
]
```

- `score`: 推薦枠は `1 - cosine_distance`（0〜1）、人気枠は `total_likes`、探索枠は `null`。
- `model_version`: `video_embeddings.model_version`。探索/人気枠は存在しない場合 `null`。
- `source`: `exploitation` / `popularity` / `exploration`。

## 利用する RPC

### `get_videos_recommendations(user_uuid, page_limit)`
- 入力: ユーザー UUID と候補数。
- 出力: 推薦候補（score, model_versionを含む）。
- Two-Tower 埋め込みが存在しない場合は空リスト（呼び出し側で探索枠にフォールバック）。

### `get_popular_videos(limit_count, lookback_days)`
- 入力: 取得件数と集計期間（日）。
- 出力: `video_popularity_daily` を元にした人気作品。
- マテビューが未リフレッシュの場合は `object_not_in_prerequisite_state` を捕捉して空リストを返す実装になっている。

### `get_videos_feed(page_limit)`
- 既存のランダム要素。`sample_video_url` が存在し過去にユーザー判断のない作品からランダム取得。
- 本関数では探索枠の候補として利用。

## スロット割り当ての詳細

1. **exploitation**  
   - ratio: 0.6  
   - 取得: `limit * CANDIDATE_MULTIPLIER`（既定3倍、最低100）で候補を取り、ランダムに抽出。  
   - score/model_version を保持。

2. **popularity**  
   - ratio: 0.2  
 - `get_popular_videos`。score = 期間内の likes 合計。model_version は `video_embeddings` から引く。
  - ローカル検証では `REFRESH MATERIALIZED VIEW public.video_popularity_daily;` を実行してから呼び出す。
  - `user_uuid` を渡すと `user_video_decisions` で既に判断済みの作品を除外する。

3. **exploration**  
   - ratio: `limit - exploitation - popularity`  
   - `get_videos_feed`。score = null, model_version = 既知ならセット。

4. **不足補填**  
   - 各枠で不足した分は他の枠（exploitation → popularity → exploration）の余剰で埋める。

## チューニング指針

- **ratio 調整**: 初期値 0.6/0.2/0.2。ユーザー行動や A/B テスト結果に応じて調整可能。
- **CANDIDATE_MULTIPLIER**: 推薦結果のバラけ具合。類似作品ばかり出る場合は拡大、精度が落ちる場合は縮小。
- **popularity lookback**: 新作を押したいときは小さく、定番を混ぜたいときは大きく。オン/オフの比較も推奨。
- **ログ/メトリクス**: `source` 別の CTR, LIKE率, 滞在時間、タグ coverage などを計測してバランス調整。

## 運用メモ

- Street 行き（本番）にデプロイする際は `supabase functions deploy videos-feed`。  
- 新マイグレーション（`get_popular_videos`, `get_videos_recommendations` 更新, `model_version` 列追加）をリモート DB に適用すること。  
- 2025-12-01 版マイグレーション（`20251201090000_fix_get_videos_recommendations_halfvec.sql`）で `halfvec(128)` 対応の `get_videos_recommendations` を再生成する。`halfvec` 化後に古い `vector(256)` の定義が残っていると `videos-feed` では常に `exploration` しか返せなくなる。
- `video_popularity_daily` のマテビュー更新 (`refresh materialized view`) をスケジュールしておくと人気枠が機能する。
- 回収データを使った再学習時、探索枠や人気枠の成果を評価し次期モデルに反映させる。
- UI 側で `source` をタグ表示する、エクスプロレーション枠を説明するなどでユーザー体験を向上可能。
