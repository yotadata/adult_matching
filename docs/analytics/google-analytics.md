# Google Analytics 仕様

Google Analytics (GA4) を用いたトラッキング方針と実装手順を整理します。環境変数による計測 ID 管理と Next.js への組み込みを前提としています。

## 目的

- プロダクトの利用状況（PV、滞在時間、遷移率など）を把握する。
- コンテンツ別の成果や推薦機能の改善余地を分析する。
- 導線の AB テストや UI 改修の効果測定に備える。

## 実装概要

- 計測対象: すべてのフロントエンドページ（ページビュー）。  
  カスタムイベントは `/swipe` 画面からのみ送信し、AI レコメンド画面や `ai-recommend` API は対象外。
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

## カスタムイベント設計（スワイプ画面）

カスタムイベントの対象は `/swipe` 画面のみです。AIレコメンド画面や `ai-recommend` API 経由の計測は行っていません。

実装ファイル: `frontend/src/app/swipe/page.tsx`（イベント送信）, `frontend/src/lib/analytics.ts`（`trackEvent` ヘルパー）。

| イベント名 | 発火タイミング | パラメータ（主要項目） | 備考 |
| ---------- | -------------- | ---------------------- | ---- |
| `recommend_fetch` | スワイプ画面が Supabase Edge Function `videos-feed` を呼び出した結果（成功/失敗） | `status` (`success`/`error`), `source`（常に `videos_feed`）, `response_ms`, `has_session`, 成功時: `videos_count`, `swipes_until_next_embed`, `decision_count` / 失敗時: `error_message` | 推薦データ取得の健全性を監視 |
| `recommend_session_start` | カードがアクティブになりセッションを開始したとき | `session_id`, `video_id`, `position`, `has_session`, `source`, `session_started_at`, `recommendation_score`, `recommendation_model_version` | `session_id` は `generateSessionId()` で生成 |
| `recommend_sample_play` | サンプル/埋め込み動画を再生開始したとき | `session_id`, `video_id`, `position`, `has_session`, `source`, `sample_type` (`sample`/`embed`), `session_started_at`, `sample_play_at`, `play_count_in_session`, `surface` (`desktop`/`mobile`) | セッション中の再生回数をカウント |
| `recommend_decision` | LIKE / NOPE の判断操作を行ったとき | `session_id`, `video_id`, `position`, `has_session`, `source`, `decision_type` (`like`/`nope`), `session_started_at`, `decision_at`, `sample_played` (0/1), `sample_last_play_at`, `sample_play_count` | 判断時点を記録 |
| `recommend_session_complete` | 判断操作後にセッションを完了したとき | `session_id`, `video_id`, `position`, `has_session`, `source`, `decision_type`, `session_started_at`, `session_completed_at`, `sample_played`, `sample_last_play_at`, `sample_play_count` | 完了時刻を保持 |
| `recommend_session_abandon` | セッション完了前に別カードへ遷移/離脱したとき | `session_id`, `video_id`, `position`, `has_session`, `source`, `session_started_at`, `session_abandoned_at`, `sample_played`, `sample_play_count` | `session_completed_at` の代わりに離脱時刻を送信 |

#### AIレコメンド画面（`/ai-recommend`）

| イベント名 | 発火タイミング | パラメータ（主要項目） | 備考 |
| ---------- | -------------- | ---------------------- | ---- |
| `ai_rec_mode_switch` | モードプリセットを切り替えたとき | `mode_id`, `tone` | 画面上部プリセットの利用頻度を把握 |
| `ai_rec_custom_apply` | カスタムモードを適用したとき | `mode_id`, `duration`, `mood`, `context` | カスタム操作の浸透度を把握 |
| `ai_rec_reason_expand` | 推薦理由を展開/閉じたとき | `mode_id`, `section_id`, `video_id`, `expanded` | 説明データ閲覧率を把握 |

### レポート観点

- `session_started_at` / `sample_play_at` / `decision_at` / `session_completed_at` / `session_abandoned_at` を利用し、滞在時間や完了率を算出する。
- `sample_played`・`sample_play_count` を用いることで、サンプル視聴有無と判断の関係を切り分けられる。
- `has_session` は Supabase 認証済みかを示し、ゲスト利用との比較に用いる。

> **注意:** PII（個人が特定できる情報）は GA4 に送信しない。必要に応じて識別子をハッシュ化し、GA4 の User-ID を利用する場合は運用ルールを整備する。

## 計測イベント追加手順

1. 追加したいユーザー行動（例: スワイプ中の新しいインタラクション）を定義する。
2. コンポーネント内で `trackEvent('<event_name>', { ...params })` を呼び出す。例:
   ```ts
   trackEvent('recommend_custom_event', {
     session_id: sessionIdRef.current,
     video_id: card.id,
     has_session: isLoggedIn,
   });
   ```
3. 追加したイベント名・パラメータを GA4 のカスタムディメンション／レポートに設定する。
4. 新イベントを導入したら必ず以下を記録する:
   - イベント名と意味、送信条件
   - パラメータのキーと型
   - 実装ファイル
   - 想定ダッシュボード・レポート

> **メモ:** 現状のカスタムイベントはすべて `/swipe` 画面から送信されます。AI レコメンド画面や API 呼び出しでは送信していない点に留意してください。

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
