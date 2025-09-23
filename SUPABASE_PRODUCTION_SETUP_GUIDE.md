# 🚀 Supabase 商用環境セットアップガイド

## 📋 作成済みファイル

✅ **設定ファイル作成完了**:
- `.env.production` - 商用環境変数ファイル
- `supabase/config.production.toml` - Supabase CLI商用設定

## 🔑 必要な情報取得手順

### Step 1: Supabase Dashboard から API Keys 取得

**URL**: `https://supabase.com/dashboard/project/mfleexehdteobgsyokex`

1. **Settings** → **API** に移動
2. 以下の値をコピー:
   ```
   Project URL: https://mfleexehdteobgsyokex.supabase.co
   anon public key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   service_role secret key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

### Step 2: データベース接続情報取得

1. **Settings** → **Database** → **Connection string**
2. URI形式をコピー:
   ```
   postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres
   ```

### Step 3: 環境変数ファイル編集

`.env.production` ファイルの以下の値を実際の値に置換:

```bash
# 1. Supabase接続情報
SUPABASE_ANON_KEY=YOUR_ANON_KEY_HERE
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY_HERE
DATABASE_URL=postgresql://postgres.[プロジェクトref]:[パスワード]@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres

# 2. Next.js用
NEXT_PUBLIC_SUPABASE_ANON_KEY=YOUR_ANON_KEY_HERE

# 3. セキュリティ
JWT_SECRET=YOUR_JWT_SECRET_HERE

# 4. ドメイン設定
NEXT_PUBLIC_SITE_URL=https://your-domain.com
```

## 🛠️ 接続手順

### 方法1: CLI アップデート後の接続（推奨）

```bash
# CLI更新後
supabase auth login
supabase link --project-ref mfleexehdteobgsyokex
```

### 方法2: 環境変数による接続

```bash
# アクセストークン設定
export SUPABASE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN

# プロジェクト接続
supabase link --project-ref mfleexehdteobgsyokex
```

### 方法3: 設定ファイル指定

```bash
# 商用設定ファイル使用
supabase --config-path ./supabase/config.production.toml status
```

## 📦 デプロイメント手順

### 1. データベースマイグレーション
```bash
supabase db push
```

### 2. Edge Functions デプロイ
```bash
supabase functions deploy enhanced_two_tower_recommendations
supabase functions deploy user-management-likes
supabase functions deploy user-management-embeddings
supabase functions deploy user-management-account
supabase functions deploy content-feed
```

### 3. シークレット設定
```bash
supabase secrets set DMM_API_ID=W63Kd4A4ym2DaycFcXSU
supabase secrets set DMM_AFFILIATE_ID=yotadata2-990
```

## 🔐 セキュリティチェックリスト

- [ ] `.env.production` を `.gitignore` に追加
- [ ] `service_role_key` の安全な管理
- [ ] 強力なJWT Secret生成
- [ ] IP制限設定（必要に応じて）
- [ ] CAPTCHA設定（hCaptcha推奨）
- [ ] SMTP設定（SendGrid推奨）

## ⚠️ 重要な注意事項

1. **環境変数の管理**
   - `.env.production` は絶対にGitにコミットしない
   - `service_role_key` は特に機密情報

2. **ドメイン設定**
   - 実際のドメインに変更が必要
   - CORS設定も実際のドメインに合わせる

3. **セキュリティ設定**
   - 商用環境では厳格なレート制限
   - MFA有効化推奨
   - CAPTCHA有効化推奨

## 🎯 次のステップ

1. **CLI更新完了後**、上記の接続手順を実行
2. **API Keys取得後**、環境変数ファイルを編集
3. **接続成功後**、マイグレーション適用
4. **Edge Functions**をデプロイ
5. **動作確認**とテスト実行

これで商用環境への接続準備が完了します！