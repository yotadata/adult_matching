# Adult Matching Backend Monitoring

リファクタリングされたバックエンドシステムの包括的監視システム

## 🎯 概要

このモニタリングシステムは、リファクタリング後のAdult Matchingバックエンドの健全性、パフォーマンス、統合性を監視します。

## 🏗️ アーキテクチャ

### コンポーネント構成

```
backend/monitoring/
├── system_monitor.py      # システムリソース監視
├── integration_monitor.py # 統合監視システム
├── __init__.py           # パッケージ初期化
└── README.md            # このファイル

supabase/functions/_shared/monitoring/
└── logger.ts            # Edge Functions用ログ・メトリクス
```

### 監視レイヤー

1. **システムレイヤー** (`SystemMonitor`)
   - CPU、メモリ、ディスク使用率
   - データベース接続数
   - プロセス監視

2. **統合レイヤー** (`IntegrationMonitor`)
   - Edge Functions ↔ Backend Services統合
   - MLパイプライン統合
   - データフロー監視

3. **アプリケーションレイヤー** (Edge Functions Logger)
   - リクエスト/レスポンス監視
   - ビジネスメトリクス
   - エラー追跡

## 🚀 使用方法

### Python Backend Monitoring

```python
import asyncio
from backend.monitoring import create_unified_monitor

async def main():
    # 統合監視システムの作成
    system_monitor, integration_monitor = create_unified_monitor()
    
    # 監視開始
    await asyncio.gather(
        system_monitor.start_monitoring(),
        integration_monitor.start_integration_monitoring()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Edge Functions Monitoring

```typescript
import { createLogger, withMonitoring } from '../_shared/monitoring/logger.ts';

// 基本的な使用
export default withMonitoring('enhanced_two_tower_recommendations')(
  async (req: Request): Promise<Response> => {
    const logger = createLogger('enhanced_two_tower_recommendations');
    
    // ユーザーコンテキスト設定
    logger.setUserContext(userId);
    
    // MLオペレーション監視
    const recommendations = await logger.monitorMLOperation(
      'generate_recommendations',
      () => generateRecommendations(userId),
      'two_tower_768'
    );
    
    return new Response(JSON.stringify(recommendations));
  }
);
```

### Configuration

```python
config = {
    'collection_interval': 15,  # seconds
    'prometheus_port': 8000,
    'alert_thresholds': {
        'cpu_critical': 90.0,
        'memory_critical': 95.0,
        'response_time_critical': 5000,  # ms
    },
    'endpoints': {
        'enhanced_recommendations': 'http://localhost:54321/functions/v1/enhanced_two_tower_recommendations',
        'data_sync': 'http://localhost:8002/health',
    }
}
```

## 📊 メトリクス

### Prometheus メトリクス

#### システムメトリクス
- `adult_matching_system_cpu_percent` - CPU使用率
- `adult_matching_system_memory_percent` - メモリ使用率
- `adult_matching_system_disk_percent` - ディスク使用率
- `adult_matching_db_connections` - データベース接続数

#### アプリケーションメトリクス
- `adult_matching_requests_total` - リクエスト総数
- `adult_matching_request_duration_seconds` - リクエスト処理時間
- `adult_matching_ml_model_accuracy` - MLモデル精度
- `adult_matching_recommendation_latency_seconds` - 推薦レイテンシ

#### Edge Functions メトリクス
- `adult_matching_edge_requests_total` - Edge Function リクエスト数
- `adult_matching_edge_*_duration_ms` - 各操作の処理時間
- `adult_matching_edge_errors_total` - エラー総数

### ログ形式

```json
{
  "level": "INFO",
  "message": "Request completed successfully",
  "data": {
    "method": "POST",
    "url": "/enhanced_two_tower_recommendations"
  },
  "context": {
    "requestId": "uuid-here",
    "userId": "user123",
    "functionName": "enhanced_two_tower_recommendations",
    "timestamp": "2025-09-16T12:30:00.000Z"
  }
}
```

## 🚨 アラート

### アラートルール

既存のPrometheusアラートルール (`backend/scripts/deployment/monitoring/alert_rules.yml`) に統合されています。

#### システムアラート
- CPU使用率 > 85% (警告), > 90% (重要)
- メモリ使用率 > 80% (警告), > 95% (重要)
- ディスク使用率 > 85% (警告), > 95% (重要)

#### アプリケーションアラート
- API応答時間 > 2秒 (警告), > 5秒 (重要)
- エラー率 > 5% (警告), > 10% (重要)
- MLモデル読み込み失敗 (重要)

#### 統合アラート
- Edge Functions ↔ Backend 統合失敗 (重要)
- データフロー停止 > 2時間 (警告)
- E2Eテスト失敗 (重要)

## 🔧 設定

### 環境変数

```bash
# Prometheus設定
PROMETHEUS_PORT=8000

# ログレベル
LOG_LEVEL=INFO

# 監視間隔
MONITORING_INTERVAL=15

# アラート設定
ALERT_WEBHOOK_URL=https://hooks.slack.com/...
```

### Supabase設定

Edge Functions用の監視には、Supabaseの環境変数でログレベルを設定：

```toml
[edge_runtime.env]
ENVIRONMENT = "production"
LOG_LEVEL = "INFO"
```

## 📈 ダッシュボード

### Prometheus + Grafana

1. **システム概要ダッシュボード**
   - CPU、メモリ、ディスク使用率
   - データベース接続数
   - リクエスト率

2. **統合監視ダッシュボード**
   - Edge Functions ステータス
   - Backend Services ステータス
   - MLパイプライン ステータス
   - データフロー健全性

3. **パフォーマンスダッシュボード**
   - レスポンス時間分布
   - スループット
   - エラー率
   - MLモデル精度

## 🔍 トラブルシューティング

### よくある問題

1. **Prometheusサーバーが起動しない**
   ```bash
   # ポート競合確認
   lsof -i :8000
   
   # 設定ファイル確認
   python -c "from backend.monitoring import create_system_monitor; m = create_system_monitor(); print(m.config)"
   ```

2. **Edge Function監視が失敗する**
   ```bash
   # Supabase設定確認
   supabase functions list
   
   # ログ確認
   supabase functions logs enhanced_two_tower_recommendations
   ```

3. **アラートが送信されない**
   ```bash
   # アラート設定確認
   cat backend/scripts/deployment/monitoring/alert_rules.yml
   
   # Prometheus設定確認
   cat backend/scripts/deployment/monitoring/prometheus.yml
   ```

## 🚀 デプロイメント

### Docker

```dockerfile
FROM python:3.12-slim

COPY backend/monitoring/ /app/monitoring/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["python", "-m", "backend.monitoring"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adult-matching-monitoring
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: monitoring
        image: adult-matching/monitoring:latest
        ports:
        - containerPort: 8000
        env:
        - name: PROMETHEUS_PORT
          value: "8000"
```

## 📝 開発・拡張

### 新しいメトリクスの追加

```python
# system_monitor.py に追加
NEW_METRIC = Gauge('adult_matching_new_metric', 'Description of new metric')

# 収集ロジック
def collect_new_metric(self):
    value = self.get_new_metric_value()
    NEW_METRIC.set(value)
```

### 新しいアラートの追加

```yaml
# alert_rules.yml に追加
- alert: NewAlert
  expr: adult_matching_new_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "New alert triggered"
```

## 🔐 セキュリティ

- メトリクスエンドポイントは内部ネットワークのみアクセス可能
- ログには機密情報を含めない
- アラート通知にはセンシティブデータを含めない
- Edge Functions認証キーは環境変数で管理

---

**Status**: ✅ Task 25 実装完了  
**Version**: 1.0.0  
**Last Updated**: 2025-09-16