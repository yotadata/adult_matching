# Google Analytics 仕様

Google Analytics (GA4) を用いたトラッキング方針と実装手順を整理します。環境変数による計測 ID 管理と Next.js への組み込みを前提としています。

## 目的

- プロダクトの利用状況（PV、滞在時間、遷移率など）を把握する。
- コンテンツ別の成果や推薦機能の改善余地を分析する。
- 導線の AB テストや UI 改修の効果測定に備える。

## 実装概要

- 計測対象: すべてのフロントエンドページ。
- 実装位置: `frontend/src/app/layout.tsx` (`RootLayout`) に gtag.js を埋め込み。
  - `Script` コンポーネントで `https://www.googletagmanager.com/gtag/js` を読込み。
  - `window.dataLayer` 初期化後に `gtag('config', '<MEASUREMENT_ID>')` を呼び出し、自動ページビュー送信を有効化。
- 実装は `NEXT_PUBLIC_GA_MEASUREMENT_ID` が設定されている場合のみ計測しています（未設定ならスクリプトは空文字を参照し、実質無効化）。

## 環境変数

| 変数名 | 説明 | 設定場所 |
| ------ | ---- | -------- |
| `NEXT_PUBLIC_GA_MEASUREMENT_ID` | GA4 の測定 ID (`G-XXXXXXX`) | `docker/env/*.env`, Vercel/Supabase などのホスティング側環境変数 |

- 本番例: `docker/env/prd.env.example` にキーのみ定義されています (`docker/env/prd.env.example:23`)。実運用では実際の測定 ID をセットしてください。
- ローカル/検証環境で GA を無効化したい場合は空文字のままにします。

## デプロイ・実行時挙動

- Docker Compose/本番環境で `NEXT_PUBLIC_GA_MEASUREMENT_ID` が設定されると、SSR/SSG の HTML に gtag スクリプトが含まれます。
- Edge Function やバックエンドには GA 組み込み不要です。計測はフロントエンドのみで完結します。
- Supabase ホスティングの場合はプロジェクト設定から該当環境変数を登録・管理します。

## 推奨カスタムイベント

ユーザー行動の把握と推薦アルゴリズム検証のため、以下のイベント送信を推奨します。イベント名は GA4 側で管理しやすいスネークケースを採用します。

| イベント名 | 発火タイミング | 主なパラメータ | 想定用途 |
| ---------- | -------------- | -------------- | -------- |
| `recommend_fetch` | `ai-recommend` / `videos-feed` API 呼び出し完了時（成功/失敗） | `status` (`success`/`error`), `response_ms`, `videos_count`, `swipes_until_next_embed`, `decision_count`, `has_session` | 推薦レスポンスの健全性確認、API 遅延監視 |
| `recommend_reset` | AI レコメンド画面のリセットボタン押下 | `has_session` (bool) | 画面リフレッシュ需要の把握 |
| `video_open` | レコメンドカードの「見る」ボタン押下 | `video_id`, `source` (`personalized`/`trending`), `position`, `has_session` | クリック率計測、枠別の誘導効果比較 |
| `video_like` / `video_dislike` | LIKE/NOPE 操作時（実装済みの場合） | `video_id`, `source`, `position`, `decision_type` | 推薦精度評価、ユーザー嗜好解析 |
| `recommend_section_view` | 推薦/トレンド枠がビューポートに入った時（IntersectionObserver 等で検知） | `section` (`personalized`/`trending`), `impression_index` | インプレッション数の把握、CTR 算出の母数管理 |

> **補足:** PII（個人を特定可能な情報）は GA へ送信しないこと。`user_id` を渡す場合はハッシュ化済み識別子の利用や GA4 の User-ID 機能の運用ルールを定めてから実装してください。

### AI レコメンド画面のフロー計測

画面操作 → サンプル再生 → LIKE/NOPE 操作までの行動と滞在時間を把握するため、以下のイベント連携を推奨します。

1. **セッション開始**  
   - 推薦カードが表示されたタイミングで `recommend_session_start` を送信。  
   - パラメータ: `session_id`, `video_id`, `position`, `has_session`, `source`, `recommendation_score`, `recommendation_model_version`。  
   - 併せて `sessionStartAt = Date.now()` をメモリ保持し、同一ユーザー操作内で利用する。

2. **サンプル再生**  
   - `SwipeCard` のプレビューオーバーレイクリック時に `recommend_sample_play` を送信。  
   - パラメータ: `session_id`, `video_id`, `source`, `position`, `elapsed_ms = Date.now() - sessionStartAt`, `sample_type`（`sample`/`embed`）、`play_count_in_session`, `surface`, `has_session`。

3. **判断操作**  
   - LIKE/NOPE ボタン（またはスワイプ完了処理）で `recommend_decision` を送信。  
   - パラメータ: `session_id`, `video_id`, `decision_type` (`like`/`nope`), `source`, `position`, `elapsed_ms`, `sample_played`（セッション中に `recommend_sample_play` が送信済みかどうか）、`sample_elapsed_ms`, `sample_play_count`, `has_session`。

4. **セッション完了**  
   - 上記の判断後に `recommend_session_complete` を送信し、`session_id`, `video_id`, `decision_type`, `total_elapsed_ms = Date.now() - sessionStartAt`, `sample_play_count`, `sample_elapsed_ms`, `has_session` を含める。  
   - 判断を行わずにページを離脱した場合は `useEffect` の `return`（アンマウント時）や `visibilitychange` で `recommend_session_abandon` を送信し、`elapsed_ms`, `sample_play_count`, `has_session` などを記録する。

5. **レポート作成**  
   - GA4 では `elapsed_ms` / `total_elapsed_ms` をカスタムメトリクスとして登録する。  
   - 「サンプル再生実施の有無 × 判断種別」で滞在時間や完了率を比較できるよう、探索レポートを作成する。

## 計測イベント追加手順

1. 追加したいユーザー行動（例: 動画閲覧ボタン押下）を定義する。
2. コンポーネント内で `window.gtag?.('event', '<event_name>', { ...params })` を呼び出す。例:
   ```ts
   if (typeof window !== 'undefined' && window.gtag) {
     window.gtag('event', 'video_open', {
       video_id: v.id,
       source: 'ai_recommend_personalized',
     });
   }
   ```
3. 追加したイベント名・パラメータを GA4 のカスタムディメンション／レポートに設定する。
4. 新イベントを導入したら必ず以下を記録する:
   - イベント名と意味、送信条件
   - パラメータのキーと型
   - 実装ファイル
   - 想定ダッシュボード・レポート

> **メモ:** 現状は自動ページビュー以外のカスタムイベントは未実装です。上記手順で必要に応じて追記してください。

## 動作確認

- GA4 管理画面の「DebugView」または「リアルタイムレポート」でイベントが届いているかを確認します。
- ブラウザで `?gtm_debug=x` を付与する、もしくは GA デバッグ用拡張機能を用いると検証が容易です。
- ページに gtag が読み込まれているかはブラウザ開発者ツールで `<script src="https://www.googletagmanager.com/gtag/js?id=...">` が存在するかを確認します。

## プライバシーと同意

- Cookie バナーやトラッキング同意の要否は運用ポリシーに従って実装してください。GDPR/個人情報保護法への対応が必要な場合は、同意管理コンポーネントを導入し、同意取得後に gtag を初期化する実装が必要です。
- ユーザーがトラッキング拒否した場合に備え、 `gtag('consent', 'update', { ad_storage: 'denied', analytics_storage: 'denied' })` を呼び出す設計も検討してください。

## 運用メモ

- 計測 ID やタグ設定を変更した場合は Pull Request / 変更履歴に記録します。
- 重要指標（UU、セッション、コンバージョンなど）のダッシュボード URL を別途 `docs/analytics/` 配下で管理するとオンボーディングが容易になります。
- 将来的に BigQuery 連携やサーバーサイド計測を追加する場合は、本ドキュメントを更新しデータフローの図示や責任範囲を明確にしてください。
