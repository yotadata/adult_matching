# ai-recommend エッジ関数仕様

`ai-recommend` は Supabase Edge Function として実装された推薦 API であり、パーソナライズ候補とトレンド作品リストを返します。Two-Tower モデル導入前のスタブとして、最新の作品と集計済みの人気作品を返す構成になっています。

## サービス概要

- 実体: `supabase/functions/ai-recommend/index.ts`
- デプロイ単位: Supabase Edge Runtime（Docker サービス `edge-ai-recommend`）
- 参照コード: `supabase/functions/ai-recommend/index.ts`

## エンドポイント

| メソッド | パス | 説明 |
| -------- | ---- | ---- |
| `OPTIONS` | `/functions/v1/ai-recommend` | CORS プレフライト。常に `200 ok`。 |
| `GET` | `/functions/v1/ai-recommend` | 推薦データ取得。ボディなし。 |
| `POST` | `/functions/v1/ai-recommend` | 同上。将来のリクエストペイロード追加を見越した受け口。 |

- 上記パスは Kong ゲートウェイ経由で公開されており、Docker Compose の `edge-ai-recommend` サービスがポート `9001` で稼働します。
- `GET` / `POST` 以外を受けると `405 Method Not Allowed` を返します。

## 認証・認可

- `Authorization` ヘッダー（`Bearer <JWT>`）を Supabase JS クライアントへそのまま渡して実行します。
- Edge Runtime 側では `VERIFY_JWT=false` のため署名検証を強制しませんが、JWT を付与すると RLS が適用された状態でクエリが走ります。
- 未ログイン時は匿名アクセス扱いとなり、公開テーブルに対してのみクエリが可能です。

## リクエスト仕様

- クエリパラメータ・ボディは現状利用しません。
- 例: `curl https://<project>.supabase.co/functions/v1/ai-recommend -H "Authorization: Bearer <token>"`.

## レスポンス仕様

### 正常系

```json
{
  "personalized": [
    {
      "id": "08e8e8ff-2ec4-4106-b997-39b0fea79e91",
      "title": "作品タイトル",
      "description": "...",
      "external_id": "dmm_12345",
      "product_url": "https://...",
      "thumbnail_url": "https://...",
      "created_at": "2024-03-18T12:34:56.000Z"
    }
  ],
  "trending": [
    {
      "id": "2d3e96db-66dc-4e39-bc1c-0aa22a4f5822",
      "title": "人気作品",
      "description": null,
      "external_id": null,
      "product_url": null,
      "thumbnail_url": null,
      "created_at": "2024-03-17T09:21:00.000Z"
    }
  ]
}
```

- `personalized` 配列: 現在は最新 10 件を返すスタブです（将来 Two-Tower 推薦に差し替え予定）。
- `trending` 配列: 直近 3 日間の `video_popularity_daily` 集計結果から likes 件数トップ 10 を返し、ランキング順を維持します。
- いずれの配列も空配列になる場合があります。トレンド計算結果が空のときは `personalized` の中身でフォールバックします。

### エラー系

- 内部クエリ失敗時は `500` とともに `{"error": "<message>"}` を返します。
- 典型例: `video_popularity_daily` の取得失敗、`videos` テーブル照会失敗など。

## 内部処理フロー

1. `Authorization` ヘッダーを Supabase クライアントへ伝播。
2. `videos` テーブルから `created_at` 降順で 10 件取得し、`personalized` として利用。
3. 現在の UTC 日付を起点に直近 3 日分の `video_popularity_daily` を取得し、likes 合計で降順ソートして上位 10 件の `video_id` を抽出。
4. 抽出した `video_id` を `videos` テーブルに問い合わせ、ランキング順に再整列して `trending` に格納。
5. トレンド候補が空の場合は `latestVideos`（= `personalized`）で代替。

## 依存データソース

- `public.videos`: 作品メタデータ。
- `public.video_popularity_daily`: likes 件数を日次集計したマテリアライズドビュー。
- `public.user_video_decisions`: likes/nope 判定を保持。`video_popularity_daily` が likes 集計元として参照。
- Supabase 認証ユーザー情報（JWT が添付される場合）。

## 実行環境

- Docker Compose の `edge-ai-recommend` コンテナで `supabase/edge-runtime:v1.69.11` イメージを利用。
- 必須環境変数
  - `SUPABASE_URL`: Kong 経由の Supabase API ベース URL。
  - `SUPABASE_ANON_KEY`: 公開キー。Edge Function から Supabase へアクセスする際に使用。
- `supabase/functions` ディレクトリが read-only でマウントされ、`/home/deno/functions/ai-recommend` がエントリポイントです。

## フロントエンド連携

- `useAiRecommend` フックが Supabase JS SDK (`supabase.functions.invoke('ai-recommend')`) で呼び出し、`personalized` / `trending` をそのまま UI に描画します。
- 呼び出し前に `supabase.auth.getSession()` を行い、セッションが存在する場合のみ `Authorization` ヘッダーを付与します。

## 今後の拡張予定

- Two-Tower 推薦モデルのエッジ推論導入に合わせて `personalized` 部分をベクトル近傍検索へ置き換える。
- `get_popular_videos(user_uuid, limit, lookback_days)` RPC を活用し、ユーザーが既に評価した作品をトレンドから除外する。
- モデルバージョンやスコアをレスポンスに含め、A/B テストや検証を容易にする。

## 運用メモ

- `video_popularity_daily` マテビューを定期的に `REFRESH MATERIALIZED VIEW` する必要があります。更新が滞るとトレンド枠が空になるため注意してください。
- ローカル検証時は `docker compose -f docker/compose.yml up edge-ai-recommend` でコンテナを起動し、Kong ルーティング経由でアクセスします。
