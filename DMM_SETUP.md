# DMM Affiliate API 設定ガイド

## 必要な環境変数

### ローカル開発用 (.env.local)
```bash
# DMM Affiliate API認証情報
DMM_API_ID="your_dmm_api_id_here"
DMM_AFFILIATE_ID="your_dmm_affiliate_id_here"
```

### Supabase本番環境用
Supabase Dashboard > Settings > Secrets で以下を設定：
```
DMM_API_ID
DMM_AFFILIATE_ID
```

### GitHub Actions用 
Repository Settings > Secrets and variables > Actions で以下を設定：
```
SUPABASE_SERVICE_ROLE_KEY
DMM_API_ID (optional, if different from Supabase secrets)
DMM_AFFILIATE_ID (optional, if different from Supabase secrets)
```

## DMM Affiliate API 取得方法

1. [DMM Affiliate Program](https://affiliate.dmm.com/) にアカウント登録
2. API利用申請を行う
3. API ID と Affiliate ID を取得
4. 上記の環境変数に設定

## 実行方法

### 1. ローカルテスト実行
```bash
cd /home/devel/dev/adult_matching
deno run --allow-net --allow-env scripts/sync-dmm.ts
```

### 2. 本番API直接呼び出し
```bash
curl -X POST "https://mfleexehdterobgsyokex.supabase.co/functions/v1/dmm_sync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SERVICE_ROLE_KEY" \
  -d '{"page": 1, "limit": 100}'
```

### 3. Supabase Cronの有効化
```sql
-- Supabase Dashboardのデータベースエディタで実行
select cron.schedule(
  'dmm-sync-daily',
  '0 2 * * *',
  $$ 
  select net.http_post(
    url:='https://your-project-url.supabase.co/functions/v1/dmm_sync',
    headers:='{"Content-Type": "application/json", "Authorization": "Bearer ' || current_setting('app.service_role_key') || '"}'::jsonb,
    body:='{"page": 1, "limit": 100}'::jsonb
  );
  $$
);
```

## 監視とログ

- Supabase Dashboard > Logs でEdge Function実行ログを確認
- GitHub Actions > Actions タブで実行履歴を確認  
- エラー時の通知設定はSlack webhookやメール通知を追加可能

## 注意事項

- DMM API のレート制限に注意（通常は1秒間に1リクエスト程度が推奨）
- 大量データ取得時は`limit`パラメータで調整
- 初回実行時は少数（limit=10）でテストすることを推奨