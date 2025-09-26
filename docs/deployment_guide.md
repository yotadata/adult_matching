# Two-Tower モデル デプロイメントガイド

## 概要
このガイドでは、Adult Video Matching ApplicationのTwo-Towerモデルを本番環境にデプロイする手順を説明します。

## 前提条件

### システム要件
- **Python 3.10+** - モデル学習とバッチ処理用
- **PostgreSQL 14+** - pgvector拡張機能付き
- **Supabase CLI** - Edge Functions管理用
- **GPU (推奨)** - モデル学習用 (NVIDIA GTX 1060以上)
- **メモリ** - 最低16GB RAM (学習時は32GB推奨)

### 依存関係
```bash
# システムパッケージ (Ubuntu/Debian)
sudo apt update
sudo apt install python3-full python3-dev postgresql-client

# Python仮想環境作成
python3 -m venv venv
source venv/bin/activate

# Python依存関係インストール
pip install -r scripts/requirements.txt
```

## 段階的デプロイメント

### Phase 1: ローカル環境での学習とテスト

#### 1.1 データベース準備
```bash
# Supabase開始
supabase start

# マイグレーション適用
supabase db reset

# データ確認
supabase dashboard
```

#### 1.2 モデル学習実行
```bash
# 学習スクリプト実行
cd scripts/
python train_two_tower_model.py \
  --db-url "postgresql://postgres:postgres@127.0.0.1:54322/postgres" \
  --embedding-dim 64 \
  --batch-size 512 \
  --epochs 20 \
  --output-dir ../models \
  --update-db
```

#### 1.3 バッチ更新テスト
```bash
# バッチエンベディング更新テスト
python batch_embedding_update.py \
  --db-url "postgresql://postgres:postgres@127.0.0.1:54322/postgres" \
  --model-path ../models \
  --dry-run
```

### Phase 2: Edge Functions デプロイ

#### 2.1 関数デプロイ
```bash
# Two-Tower推奨機能デプロイ
supabase functions deploy two_tower_recommendations

# デプロイ確認
supabase functions list
```

#### 2.2 テスト実行
```bash
# 認証付きテスト
curl -X POST "https://your-project.supabase.co/functions/v1/two_tower_recommendations" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "exclude_liked": true}'
```

### Phase 3: 本番環境セットアップ

#### 3.1 環境変数設定
```bash
# Supabase環境変数
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"

# データベース接続
export DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@db.your-project.supabase.co:5432/postgres"
```

#### 3.2 本番データベース設定
```sql
-- pgvector拡張確認
CREATE EXTENSION IF NOT EXISTS vector;

-- インデックス作成
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_video_embeddings_updated_at 
ON video_embeddings(updated_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_embeddings_updated_at 
ON user_embeddings(updated_at);

-- RLS確認
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN ('video_embeddings', 'user_embeddings');
```

#### 3.3 モデルファイルアップロード
```bash
# Supabase Storage バケット作成
supabase storage create --name models --public false

# モデルファイルアップロード
supabase storage upload models/user_tower.onnx ./models/user_tower.onnx
supabase storage upload models/item_tower.onnx ./models/item_tower.onnx
supabase storage upload models/preprocessing.pkl ./models/preprocessing.pkl
```

### Phase 4: 自動化とモニタリング

#### 4.1 Cronジョブ設定
```bash
# crontab編集
crontab -e

# 毎日深夜2時にバッチ更新実行
0 2 * * * /path/to/project/scripts/run_batch_update.sh >> /var/log/two_tower_batch.log 2>&1

# 週次フル再学習 (日曜日午前3時)
0 3 * * 0 /path/to/project/scripts/full_retrain.sh >> /var/log/two_tower_retrain.log 2>&1
```

#### 4.2 監視設定
```bash
# ログローテーション設定
sudo tee /etc/logrotate.d/two_tower << EOF
/var/log/two_tower*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

## パフォーマンス最適化

### データベース最適化
```sql
-- 定期的なVACUUM実行
VACUUM ANALYZE video_embeddings;
VACUUM ANALYZE user_embeddings;

-- 統計情報更新
ANALYZE video_embeddings;
ANALYZE user_embeddings;

-- 不要な古いデータクリーンアップ
DELETE FROM user_embeddings 
WHERE updated_at < NOW() - INTERVAL '30 days'
AND user_id NOT IN (
    SELECT user_id FROM likes 
    WHERE created_at > NOW() - INTERVAL '7 days'
);
```

### Edge Functions最適化
```typescript
// メモリ使用量削減のため
const BATCH_SIZE = 100;  // 一度に処理する動画数を制限
const CACHE_DURATION = 300; // 5分間キャッシュ

// レスポンス時間監視
console.time('two_tower_inference');
// ... 処理 ...
console.timeEnd('two_tower_inference');
```

## トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー
```bash
# 解決策: バッチサイズ削減
python train_two_tower_model.py --batch-size 256  # デフォルト512から削減
```

#### 2. pgvector次元数エラー
```sql
-- 確認方法
SELECT vector_dims(embedding) FROM video_embeddings LIMIT 1;

-- 修正: embedding_dim設定確認
python train_two_tower_model.py --embedding-dim 64
```

#### 3. Edge Functions timeout
```typescript
// タイムアウト設定調整
const TIMEOUT_MS = 30000;  // 30秒

// 処理時間監視
if (Date.now() - startTime > TIMEOUT_MS) {
    return earlyResponse;
}
```

#### 4. 推奨品質の問題
```bash
# 学習データの確認
SELECT COUNT(*) FROM likes WHERE created_at > NOW() - INTERVAL '6 months';
SELECT COUNT(DISTINCT user_id) FROM likes;
SELECT COUNT(DISTINCT video_id) FROM likes;

# 最小要件: 10万いいね, 1000ユーザー, 10000動画
```

### ログ監視コマンド
```bash
# リアルタイムログ監視
tail -f /var/log/two_tower_batch.log

# エラーのみ抽出
grep -E "(ERROR|CRITICAL)" /var/log/two_tower*.log

# パフォーマンス監視
grep "completed successfully" /var/log/two_tower_batch.log | tail -10
```

## セキュリティ考慮事項

### データ保護
- モデルファイルは非公開ストレージに保存
- データベース接続は暗号化必須
- API キーの定期ローテーション
- 個人情報の匿名化

### アクセス制御
```sql
-- RLS (Row Level Security) 確認
ALTER TABLE user_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE video_embeddings ENABLE ROW LEVEL SECURITY;

-- ユーザー自身のデータのみアクセス可能
CREATE POLICY "Users can read own embeddings" ON user_embeddings
FOR SELECT USING (auth.uid()::text = user_id);
```

## 運用メトリクス

### 監視すべき指標
- **学習精度**: AUC > 0.65, Precision@10 > 0.15
- **レスポンス時間**: 推奨API < 200ms, バッチ処理 < 30min
- **データベース**: CPU使用率 < 80%, 接続数 < 100
- **メモリ使用率**: Edge Functions < 128MB
- **エラー率**: API エラー率 < 1%

### アラート設定
```bash
# バッチ処理失敗アラート
if ! grep -q "completed successfully" /var/log/two_tower_batch.log; then
    echo "Batch update failed!" | mail -s "Two-Tower Alert" admin@example.com
fi
```

## パフォーマンス ベンチマーク

### 期待値
- **学習時間**: フルデータセット 4-6時間 (GPU使用時)
- **推論時間**: ユーザー推奨 < 200ms
- **バッチ更新**: 1000ユーザー < 10分
- **メモリ使用量**: 学習時 < 8GB, 推論時 < 1GB

### 測定コマンド
```bash
# 学習時間測定
time python train_two_tower_model.py --epochs 5

# API レスポンス時間
curl -w "%{time_total}s\\n" -X POST "https://your-api/recommendations"

# バッチ処理時間
time ./run_batch_update.sh
```

---

## 最終チェックリスト

### デプロイ前確認
- [ ] すべてのテストが pass
- [ ] 学習データの品質確認 (最低10万interactions)
- [ ] モデルの精度確認 (AUC > 0.65)
- [ ] Edge Functions の動作確認
- [ ] データベースインデックス作成
- [ ] 環境変数設定完了
- [ ] バックアップ作成

### デプロイ後確認
- [ ] API レスポンス時間 < 200ms
- [ ] エラー率 < 1%
- [ ] バッチ処理正常動作
- [ ] ログ出力確認
- [ ] モニタリング設定完了
- [ ] アラート動作確認

## サポート

問題が発生した場合は、以下の情報を収集してください：
- エラーログ
- システム情報 (CPU, メモリ使用率)
- データベース統計
- API レスポンス時間
- 学習データ統計

詳細な技術サポートについては、プロジェクトドキュメントまたは開発チームにお問い合わせください。