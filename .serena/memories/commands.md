# Development Commands

## Frontend Development (from `/frontend` directory)
```bash
# 開発サーバー起動
npm run dev              # localhost:3000で開発サーバー開始

# ビルドとデプロイ
npm run build           # 本番用ビルド生成
npm run start           # 本番サーバー起動
npm run lint            # ESLintでコードチェック

# 依存関係管理
npm install             # 依存関係インストール
npm update              # 依存関係更新
```

## Supabase Edge Functions
```bash
# Edge Functions開発
supabase functions serve                    # ローカルでEdge Functions起動
supabase functions deploy [function-name]  # 特定の関数デプロイ
supabase functions logs [function-name]    # 関数ログ確認

# データベース管理
supabase db reset                           # データベースリセット
supabase db push                            # マイグレーション適用
supabase migration new [migration-name]    # 新しいマイグレーション作成
```

## Machine Learning Scripts (from `/scripts` directory)
```bash
# Python環境セットアップ
python3 -m venv venv                       # 仮想環境作成
source venv/bin/activate                   # 仮想環境アクティベート
pip install -r requirements.txt           # 依存関係インストール

# Two-Tower モデル学習
python train_two_tower_model.py \
  --db-url $DATABASE_URL \
  --embedding-dim 768 \
  --epochs 20 \
  --update-db

# 学習実行例
./train_example.sh                         # サンプル学習スクリプト実行
```

## API Testing
```bash
# Edge Functions テスト (例: recommendations)
curl -X POST $SUPABASE_URL/functions/v1/recommendations \
  -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "exclude_liked": true}'

# バッチエンベディング更新
curl -X POST $SUPABASE_URL/functions/v1/update_embeddings \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
  -d '{"update_type": "both", "batch_size": 100}'
```

## Git Workflow
```bash
# 標準的な開発フロー
git status                                 # 変更状況確認
git add .                                  # ステージング
git commit -m "feat: add new feature"     # コミット
git push origin feature-branch            # プッシュ

# ブランチ管理
git checkout -b feature/new-feature        # 新機能ブランチ作成
git merge main                             # メインブランチをマージ
```

## Environment Variables Required
```bash
# Frontend (.env.local)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key

# Scripts (.env)
DATABASE_URL=postgresql://user:pass@host:port/dbname
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

## Task Completion Workflow
1. **Code Development**: 機能実装・修正
2. **Linting**: `npm run lint` でコードチェック
3. **Testing**: 手動テスト・動作確認
4. **Build Verification**: `npm run build` で本番ビルド確認
5. **Git Commit**: 変更をコミット
6. **Deploy**: Supabase/Vercelにデプロイ (必要に応じて)