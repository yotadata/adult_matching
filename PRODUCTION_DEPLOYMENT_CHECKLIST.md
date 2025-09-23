# 🚀 Supabase本番環境デプロイメントチェックリスト

## 現在の状況分析

**現在**: ローカル開発環境でSupabaseが稼働中
**目標**: 実際のSupabase本番環境へのデプロイメント

## ❌ 現在不足している要素

### 1. Supabaseプロジェクト設定

#### 🔴 必須: Supabaseプロジェクト作成・設定
```bash
# 現在のステータス: ローカル開発環境のみ
# 必要: 実際のSupabaseプロジェクト

# 不足要素:
- Supabase Dashboardでの新規プロジェクト作成
- 本番用プロジェクトID
- 本番用API キー (anon_key, service_role_key)
- 本番用データベースURL
```

**対応手順**:
1. [Supabase Dashboard](https://supabase.com/dashboard) でアカウント作成
2. 新規プロジェクト作成 (名前: adult-matching-production)
3. プロジェクトキー取得

### 2. 環境変数設定

#### 🔴 必須: 本番環境変数
```bash
# 現在: 環境変数未設定
# 必要: 本番環境用環境変数

# 作成が必要な .env.production:
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-production-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-production-service-role-key

# オプション (DMM API):
DMM_API_ID=your-dmm-api-id
DMM_AFFILIATE_ID=your-dmm-affiliate-id
```

### 3. Supabase CLI設定

#### 🔴 必須: プロジェクトリンク
```bash
# 現在: ローカルプロジェクトのみ
# 必要: 本番プロジェクトとのリンク

# 実行が必要:
supabase link --project-ref your-project-ref
```

#### ⚠️ 推奨: CLI更新
```bash
# 現在: v2.34.3
# 最新: v2.40.7
# 更新推奨 (新機能・バグ修正)
```

### 4. データベース移行

#### 🔴 必須: マイグレーション適用
```bash
# 現在: ローカルDBのみ
# 必要: 本番DBへのマイグレーション適用

# 実行が必要:
supabase db push
# または
supabase db reset --linked
```

### 5. Edge Functions デプロイメント

#### 🔴 必須: Functions デプロイ
```bash
# 現在: ローカル環境のみ
# 必要: 本番環境へのデプロイ

# 実行が必要:
supabase functions deploy enhanced_two_tower_recommendations
supabase functions deploy user-management-likes
supabase functions deploy user-management-embeddings
supabase functions deploy user-management-account
supabase functions deploy content-feed
```

## ✅ 段階別デプロイメント手順

### Phase 1: Supabaseプロジェクト準備

#### Step 1.1: プロジェクト作成
1. **Supabase Dashboard アクセス**
   - https://supabase.com/dashboard
   - GitHubアカウントでサインアップ/ログイン

2. **新規プロジェクト作成**
   ```
   Project Name: adult-matching-production
   Database Password: [強力なパスワード設定]
   Region: Northeast Asia (Tokyo) - ap-northeast-1
   ```

3. **プロジェクト情報取得**
   - Project Reference ID
   - API URL
   - anon public key
   - service_role secret key

#### Step 1.2: CLI更新・設定
```bash
# CLI更新 (推奨)
npm update -g @supabase/cli

# プロジェクトリンク
supabase link --project-ref YOUR_PROJECT_REF

# 認証確認
supabase auth login
```

### Phase 2: 環境設定

#### Step 2.1: 環境変数ファイル作成
```bash
# 本番環境変数ファイル作成
cat > .env.production << EOF
# Supabase本番環境設定
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY

# オプション: DMM API
DMM_API_ID=YOUR_DMM_API_ID
DMM_AFFILIATE_ID=YOUR_DMM_AFFILIATE_ID

# 監視設定
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=production
EOF
```

#### Step 2.2: Supabase Secrets設定
```bash
# Edge Functions用シークレット設定
supabase secrets set DMM_API_ID=your_api_id
supabase secrets set DMM_AFFILIATE_ID=your_affiliate_id
supabase secrets set ENVIRONMENT=production
```

### Phase 3: データベース移行

#### Step 3.1: マイグレーション確認
```bash
# ローカルマイグレーション状態確認
supabase db diff

# 本番データベースとの差分確認
supabase db diff --linked
```

#### Step 3.2: マイグレーション適用
```bash
# 本番データベースにマイグレーション適用
supabase db push

# または完全リセット (初回のみ)
supabase db reset --linked
```

#### Step 3.3: データベース確認
```bash
# 本番データベース接続確認
supabase db shell --linked
\dt  # テーブル一覧確認
\df  # 関数一覧確認
\q   # 終了
```

### Phase 4: Edge Functions デプロイメント

#### Step 4.1: Functions個別デプロイ
```bash
# 推薦システム
supabase functions deploy enhanced_two_tower_recommendations

# ユーザー管理
supabase functions deploy user-management-likes
supabase functions deploy user-management-embeddings
supabase functions deploy user-management-account

# コンテンツ
supabase functions deploy content-feed
```

#### Step 4.2: デプロイメント確認
```bash
# 全Functions確認
supabase functions list

# 個別Function確認
supabase functions inspect enhanced_two_tower_recommendations

# ログ確認
supabase functions logs enhanced_two_tower_recommendations
```

### Phase 5: 本番動作確認

#### Step 5.1: API動作テスト
```bash
# 推薦API テスト
curl -X POST "https://YOUR_PROJECT_REF.supabase.co/functions/v1/enhanced_two_tower_recommendations" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "num_recommendations": 5}'

# コンテンツフィード テスト
curl -X POST "https://YOUR_PROJECT_REF.supabase.co/functions/v1/content-feed" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"feed_type": "explore", "limit": 10}'
```

#### Step 5.2: データベース動作確認
```bash
# RPC関数テスト
curl -X POST "https://YOUR_PROJECT_REF.supabase.co/rest/v1/rpc/get_personalized_videos_feed" \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"p_user_id": "test-user", "p_limit": 10}'
```

## 🛠️ 自動デプロイメントスクリプト

現在のファイルを本番環境用に更新し、自動デプロイメントスクリプトを作成します：

### 必要なスクリプト修正

1. **環境変数読み込み修正**
   - 各Edge Functionで本番環境変数を読み込み
   - ローカル開発環境との切り替え対応

2. **デプロイメントスクリプト作成**
   - 段階的デプロイメント
   - ロールバック機能
   - 動作確認テスト

3. **設定ファイル更新**
   - supabase/config.toml の本番環境設定
   - CI/CD パイプライン設定

## ⚠️ 重要な注意事項

### セキュリティ
- **service_role_key は絶対に公開しない**
- **環境変数ファイルは .gitignore に追加**
- **DMM API キーは Supabase Secrets で管理**

### データ
- **本番環境では初期データを慎重に投入**
- **個人情報・成人向けコンテンツの適切な取り扱い**
- **GDPR・日本の個人情報保護法への準拠**

### 監視
- **本番環境監視の設定**
- **アラート設定**
- **ログ保持期間の設定**

## 📋 実行前チェックリスト

- [ ] Supabaseアカウント作成
- [ ] 本番プロジェクト作成
- [ ] プロジェクトキー取得
- [ ] .env.production 作成
- [ ] supabase link 実行
- [ ] マイグレーション適用
- [ ] Edge Functions デプロイ
- [ ] 動作確認テスト
- [ ] 監視設定
- [ ] ドキュメント更新

## 🚀 次のステップ

実際の本番デプロイメントを実行するには：

1. **まず Supabaseプロジェクト作成** (最重要)
2. **環境変数設定**
3. **段階的デプロイメント実行**
4. **動作確認**
5. **監視開始**

これらの手順を実行すれば、実際のSupabase本番環境への完全なデプロイメントが可能になります。

---

**質問**: Supabaseプロジェクトの作成から始めますか？それとも特定のステップについて詳細な説明が必要ですか？