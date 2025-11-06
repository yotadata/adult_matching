# analysis-results エッジ関数仕様

性癖分析 API は Supabase Edge Function として実装され、ユーザーの LIKE / NOPE 履歴を集計したサマリーを返します。  
フロントエンド (`/analysis-results` 画面) から `supabase.functions.invoke('analysis-results')` で呼び出し、ダッシュボード表示に利用します。

## サービス概要

- 実体: `supabase/functions/analysis-results/index.ts`
- 公開エンドポイント: `/functions/v1/analysis-results`
- フロント側参照: `frontend/src/hooks/useAnalysisResults.ts` → `frontend/src/app/analysis-results/page.tsx`

## エンドポイント

| メソッド | パス | 説明 |
| -------- | ---- | ---- |
| `GET` | `/functions/v1/analysis-results` | 認証ユーザーのサマリーを取得 |
| `POST` | `/functions/v1/analysis-results` | 同上。`invoke`（POST）利用時はこちらを使用 |

- CORS: `Access-Control-Allow-Origin: *`、`GET/POST/OPTIONS` を許可。
- `OPTIONS` は 200 で空ボディを返却。

## 認証と権限

- Supabase Auth の JWT（`Authorization: Bearer <token>`）が必須。
- Edge Runtime では `VERIFY_JWT=false` だが、`createClient` に Authorization ヘッダーを渡して RLS を効かせる。
- 未ログイン時（`anon`）は空データを返しつつ 200 を返却。

## パラメータ

| パラメータ | 型 | 既定値 | 備考 |
| ---------- | -- | ------ | ---- |
| `window_days` | number \| `'all'` | `90` | 直近 N 日間で集計。`'all'` で全期間。 |
| `include_nope` | boolean | `false` | `true` にすると NOPE 件数をランキングにも反映する。 |
| `tag_limit` | number | `6` | タグランキングの件数（最大 20）。 |
| `performer_limit` | number | `6` | 出演者ランキングの件数（最大 20）。 |
| `recent_limit` | number | `10` | 直近判断履歴の返却件数（最大 50）。 |

例:

```
GET /functions/v1/analysis-results?window_days=30
Authorization: Bearer <JWT>
```

## レスポンス

```jsonc
{
  "summary": {
    "total_likes": 42,
    "total_nope": 18,
    "like_ratio": 0.7,
    "first_decision_at": "2024-02-01T03:21:54.231Z",
    "latest_decision_at": "2024-03-17T12:05:09.101Z",
    "window_days": 30,
    "sample_size": 60
  },
  "top_tags": [
    {
      "tag_id": "b3cf6c9f-9a91-4407-b3a9-31eabf5618a0",
      "tag_name": "巨乳",
      "likes": 12,
      "nopes": 1,
      "like_ratio": 0.92,
      "share": 0.286,
      "last_liked_at": "2024-03-16T05:41:00.201Z",
      "representative_video": {
        "id": "92d1db47-8c15-43ce-8b5b-6c51802fa1cf",
        "title": "Fカップ○○ 120分",
        "thumbnail_url": "https://...",
        "product_url": "https://..."
      }
    }
  ],
  "top_performers": [
    {
      "performer_id": "ac3f3d3b-afe0-4bcb-81b9-3fe5dbcd1f1e",
      "performer_name": "山田花子",
      "likes": 8,
      "nopes": 0,
      "like_ratio": 1.0,
      "share": 0.19,
      "last_liked_at": "2024-02-28T11:25:33.601Z",
      "representative_video": {
        "id": "f7f79c4f-1c74-4655-9744-bb846f8c3621",
        "title": "新人デビュー作",
        "thumbnail_url": "https://...",
        "product_url": "https://..."
      }
    }
  ],
  "recent_decisions": [
    {
      "video_id": "f7f79c4f-1c74-4655-9744-bb846f8c3621",
      "title": "新人デビュー作",
      "decision_type": "like",
      "decided_at": "2024-03-17T12:05:09.101Z",
      "thumbnail_url": "https://...",
      "product_url": "https://...",
      "tags": [
        {"id": "b3cf6c9f-9a91-4407-b3a9-31eabf5618a0","name": "巨乳"}
      ],
      "performers": [
        {"id": "ac3f3d3b-afe0-4bcb-81b9-3fe5dbcd1f1e","name": "山田花子"}
      ]
    }
  ]
}
```

- `top_tags[].share`: 該当タグの LIKE 数 / 全 LIKE 数。
- `top_performers[].representative_video`: その出演者を含む最新 LIKE 動画。存在しない場合は `null`。
- `recent_decisions[]`: 指定件数分を最新順で返却。タグ/出演者は各最大 6 件。

### エラー例

| ステータス | 内容 |
| ---------- | ---- |
| `401` | `{"error":"unauthorized"}` JWT 不備 |
| `500` | `{"error":"internal_error"}` 内部例外 |

## 集計ロジック

1. `user_video_decisions` を `created_at DESC` でページング取得（最大 2,000 件）。
2. `window_days` 指定時は `created_at >= now() - interval 'window_days days'` で絞り込み。
3. LIKE/NOPE 件数、最初・最新の判断日時、サンプルサイズを集計。
4. タグ／出演者のランキングをメモリ上で集計（LIKE 数、NOPE 数、最新 LIKE、代表作品）。
5. `include_nope=false` の場合も NOPE 件数は統計として保持。ただし UI での表示可否は呼び出し側に委ねる。

## 今後の拡張候補

- タグ・出演者別ランキング、直近判断履歴の復活（単一 SQL / RPC での集計）。
- Supabase RPC 化やマテビュー活用による高速化。
- フロントからのキャッシュ戦略や A/B 用の分析メトリクス追加。
