# Project Structure

## Root Directory Structure
```
adult_matching/
├── frontend/           # Next.js フロントエンドアプリケーション
├── supabase/          # Supabase Edge Functions
├── scripts/           # ML学習スクリプト、データ取り込み
├── docs/             # プロジェクトドキュメント
├── db/               # データベースマイグレーション
├── .github/          # GitHub Actions設定
└── CLAUDE.md         # プロジェクトガイダンス
```

## Frontend Structure (`/frontend`)
```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── page.tsx           # メインスワイプインターフェース
│   │   ├── layout.tsx         # ルートレイアウト
│   │   ├── globals.css        # グローバルスタイル
│   │   ├── api/auth/          # 認証API routes
│   │   ├── account-management/ # アカウント管理ページ
│   │   └── analysis-results/   # 分析結果ページ
│   ├── components/            # React components
│   │   ├── SwipeCard.tsx      # メインスワイプカード
│   │   ├── MobileVideoLayout.tsx # モバイル最適化レイアウト
│   │   ├── ActionButtons.tsx   # デスクトップ操作ボタン
│   │   ├── Header.tsx         # アプリケーションヘッダー
│   │   ├── LikedVideosDrawer.tsx # いいね一覧ドロワー
│   │   └── auth/              # 認証関連コンポーネント
│   ├── hooks/                 # カスタムHooks
│   │   └── useMediaQuery.ts   # レスポンシブ対応Hook
│   └── lib/                   # ユーティリティ
│       ├── supabase.ts        # Supabaseクライアント
│       └── supabaseAdmin.ts   # Supabase管理者クライアント
├── package.json               # 依存関係とスクリプト
├── tsconfig.json             # TypeScript設定
├── tailwind.config.js        # Tailwind CSS設定
└── eslint.config.mjs         # ESLint設定
```

## Supabase Structure (`/supabase`)
```
supabase/
├── functions/                 # Edge Functions
│   ├── feed_explore/         # 多様な動画フィード
│   ├── recommendations/      # AI推奨システム
│   ├── update_user_embedding/ # ユーザーエンベディング更新
│   ├── likes/                # いいね管理
│   ├── delete_account/       # アカウント削除
│   └── update_embeddings/    # バッチエンベディング更新
└── migrations/               # データベースマイグレーション
```

## Scripts Structure (`/scripts`)
```
scripts/
├── train_two_tower_model.py  # Two-Tower ML model training
├── train_example.sh          # 学習スクリプト実行例
├── requirements.txt          # Python dependencies
└── fanza_ingest.ts          # 動画データ取り込み
```

## Database Schema (Main Tables)
- **videos**: 動画メタデータとメディアURL
- **video_embeddings**: 動画の768次元ベクトル埋め込み
- **user_embeddings**: ユーザー嗜好ベクトル
- **likes**: ユーザーの動画評価履歴
- **tags/tag_groups**: 柔軟なタギングシステム
- **performers**: 出演者情報
- **profiles**: ユーザープロフィール

## Key Features by Directory
- **Frontend**: ユーザーインターフェース、認証、状態管理
- **Supabase**: API エンドポイント、認証、データベースアクセス
- **Scripts**: ML モデル学習、データ取り込み、バッチ処理
- **Docs**: アーキテクチャ、API、設計ドキュメント