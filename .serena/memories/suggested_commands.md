# Suggested Commands for Development

## Daily Development Workflow

### 1. 開発開始
```bash
# プロジェクトディレクトリに移動
cd /home/devel/dev/adult_matching

# 最新のコードを取得
git pull origin main

# フロントエンド開発サーバー起動
cd frontend
npm run dev

# 別ターミナルでSupabase Edge Functions起動
supabase functions serve
```

### 2. コード開発中
```bash
# リアルタイムでlinting確認
npm run lint

# TypeScriptエラーチェック
npx tsc --noEmit

# 変更をステージング
git add .
git status
```

### 3. 機能完成時
```bash
# 最終lintチェック
npm run lint

# ビルドテスト
npm run build

# コミット前の確認
git diff --cached

# コミット
git commit -m "feat: describe your changes"

# プッシュ
git push origin your-branch-name
```

## Frequently Used Commands

### Frontend Development
```bash
# 依存関係インストール
npm install

# 開発サーバー（ホットリロード）
npm run dev

# 本番ビルド
npm run build

# ESLintでコード品質チェック
npm run lint

# 新しいコンポーネント作成時のファイル確認
ls -la src/components/
```

### Supabase Management
```bash
# Edge Function開発
supabase functions serve

# 特定の関数のログ確認
supabase functions logs recommendations

# データベーススキーマ確認
supabase db diff

# 新しいマイグレーション作成
supabase migration new add_new_feature

# Edge Function デプロイ
supabase functions deploy recommendations
```

### Machine Learning Development
```bash
# Python環境準備
cd scripts
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Two-Towerモデル学習
python train_two_tower_model.py \
  --db-url $DATABASE_URL \
  --embedding-dim 768 \
  --epochs 10 \
  --update-db

# バッチエンベディング更新
curl -X POST $SUPABASE_URL/functions/v1/update_embeddings \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
  -d '{"update_type": "both"}'
```

### Debugging and Testing
```bash
# API手動テスト（recommendations）
curl -X POST localhost:54321/functions/v1/recommendations \
  -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10}'

# データベース直接クエリ（psql）
psql $DATABASE_URL -c "SELECT COUNT(*) FROM videos;"

# ログファイル確認
tail -f supabase/logs/edge-functions.log

# ネットワーク接続確認
curl -I $SUPABASE_URL/rest/v1/
```

### File Operations
```bash
# プロジェクト構造確認
tree -I node_modules

# 特定のファイル検索
find . -name "*.tsx" -type f

# 特定のテキスト検索
grep -r "CardData" frontend/src/ --include="*.ts*"

# 大きなファイル確認
du -sh frontend/node_modules
```

### Git Workflow
```bash
# 新機能開発開始
git checkout -b feature/new-recommendation-algorithm

# 作業途中保存
git add .
git commit -m "wip: working on new algorithm"

# メインブランチの最新を取得
git checkout main
git pull origin main

# フィーチャーブランチに最新をマージ
git checkout feature/new-recommendation-algorithm
git merge main

# プルリクエスト準備
git push origin feature/new-recommendation-algorithm
```

## Environment Setup Commands
```bash
# Node.js バージョン確認
node --version
npm --version

# Supabase CLI確認
supabase --version

# Python環境確認
python3 --version
pip --version

# データベース接続テスト
psql $DATABASE_URL -c "SELECT version();"
```

## Performance Monitoring
```bash
# Next.js バンドルサイズ分析
npm run build -- --analyze

# Edge Function のメモリ使用量確認
supabase functions logs --filter memory

# データベースパフォーマンス
psql $DATABASE_URL -c "SELECT * FROM pg_stat_activity;"
```