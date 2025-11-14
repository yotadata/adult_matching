# AIで探す – 機能仕様（ドラフト）

## 目的
- スワイプ前に「自分向け」「全体トレンド」「今の気分」の3視点から作品セットを提示し、AIに相談して決める体験を提供する。
- モード切り替えや複雑な意図設定を廃止し、最小の入力で多様な提案を得られるユーザー体験にする。

## ユースケース
1. ユーザーが LIKE 履歴をベースに“自分専用”のセットを確認する。
2. 今、コミュニティ全体で人気が高まっている作品を参考にする。
3. 「甘め」「刺激的」「パートナーと」など気分キーワードを入力し、即席で AI がマッチングしたセットを得る。

## 画面要件（`frontend/src/app/ai-recommend/page.tsx`）
- **ヒーロー**：タイトル・説明文・「再取得」ボタン（`refetch` 呼び出し）。
- **気分入力カード**：テキストエリア＋チップ。入力値は `prompt` として API に送る。適用/クリアで `appliedPrompt` を更新。適用時は展開中のカード state をクリア。
- **嗜好スナップショット**：直近90日の Topタグ/出演者（`useAnalysisResults`）を表示。
- **セクションリスト**：API から返る `sections` を順番に描画。各セクションにはタイトル/サブコピー/カード群（横スクロール）を表示。カードにはタグ/理由/指標/作品リンクを含む。
- **状態表示**：ローディング（骨組み3枚）、エラー（赤バナー）、候補ゼロ（空表示メッセージ）。

## バックエンド要件（`supabase/functions/ai-recommend`）
- リクエスト: `POST|GET { prompt?: string, limit_per_section?: number }`
- **セクション構成**（最低3つ）
  1. `for-you`: `get_videos_recommendations` RPC（ユーザー埋め込み）から `limit_per_section` 件。
  2. `trend-now`: `get_popular_videos` RPC（lookback 7日）から `limit_per_section` 件。
  3. `prompt-match`: ユーザー入力キーワードでタグ・出演者・タイトルを部分一致。未入力時は最新 (`videos` テーブル) から `limit_per_section` 件。
- 各候補は `hydrateVideoDetails` で product_url / preview_url / duration を補完し、`pickUnique` でセクション間の重複を排除。
- レスポンス: `{ generated_at, sections[], metadata }`。metadata には候補件数、キーワード、ユーザー文脈、セクション上限数を含める。
- fallback: すべて空の場合、トレンド+新着を混ぜた「おすすめセット」を返す。

## 主要な挙動
- **適用ボタン**: `promptInput` を `appliedPrompt` へ反映→ `useAiRecommend` を再実行。`expandedItemId` を `null` に戻す。
- **チップ追加**: 既に含まれていないキーワードのみ追記。
- **セクション描画**: `LayersIcon` でセクションIDに応じたアイコン/色を表示（for-you=緑、trend=青、prompt=ピンク）。
- **説明文**: Edge Function が `reason.summary/detail` を返し、カード内で表示する。

## 受け入れ観点
- 非ログイン時は API が 401 -> 画面にエラーを表示し、サイドバー/ヘッダーからも遷移できない（既存制御）。
- キーワードを変更すると 3 番目のセクションの `title`/`rationale` に入力語が反映される。
- セクションが 1 つでも候補を持つ限り、カードを横スクロールで閲覧できる。
- エラー/ローディング/空状態の表示が崩れない。

## 今後の拡張余地
- prompt キーワードをサジェストする自動案内。
- セクションごとの click イベントを `trackEvent` へ送信。
- `prompt-match` のマッチロジックをタグ/出演者だけでなく embedding 近傍検索へ拡張。

（最終確定前のドラフト。実装進行に合わせて更新すること）
