# 統合コンテンツAPI包括的テストスイート

## 概要

このテストスイートは、統合コンテンツAPIの全機能を包括的にテストするために作成されました。全フィードタイプ、エラーハンドリング、パフォーマンス要件、セキュリティ要件を検証します。

## テスト構成

### テストファイル

1. **`test_content_api_comprehensive.py`**
   - 基本機能テスト
   - 全フィードタイプ（explore, personalized, latest, popular, random）
   - フィルタリングとページネーション
   - 認証とセキュリティ
   - データ品質検証

2. **`test_content_api_edge_cases.py`**
   - エッジケースとエラー条件
   - 異常なリクエストパラメータ
   - セキュリティ攻撃のシミュレーション
   - 障害復旧テスト
   - ストレス条件下での動作

3. **`test_content_api_performance.py`**
   - レスポンス時間要件（<500ms）
   - スループット要件（>50 RPS）
   - 同時接続処理
   - リソース使用量監視
   - スケーラビリティテスト

### サポートファイル

- **`conftest.py`**: テスト設定とフィクスチャ
- **`__init__.py`**: モジュール初期化
- **`run_content_api_tests.py`**: 統合テストランナー

## テスト実行

### 基本実行

```bash
# 全テストの実行
python run_content_api_tests.py all

# 基本機能テストのみ
python run_content_api_tests.py comprehensive

# エッジケーステストのみ
python run_content_api_tests.py edge_cases

# パフォーマンステストのみ
python run_content_api_tests.py performance

# クイックスモークテスト
python run_content_api_tests.py smoke
```

### オプション

```bash
# 低速テストも含める
python run_content_api_tests.py all --include-slow

# 詳細出力
python run_content_api_tests.py comprehensive --verbose

# カスタムテストディレクトリ
python run_content_api_tests.py all --test-dir /path/to/tests
```

### Pytest直接実行

```bash
# 基本実行
pytest test_content_api_comprehensive.py -v

# 特定のマーカーのみ
pytest -m "content_api and not slow" -v

# パフォーマンステストのみ
pytest -m "performance" -v

# カバレッジ付き実行
pytest --cov=backend/functions --cov-report=html -v
```

## テスト要件と成功基準

### 機能要件

✅ **全フィードタイプの正常動作**
- explore: 多様性重視の探索フィード
- personalized: ユーザー埋め込みに基づくパーソナライズドフィード
- latest: 最新動画フィード
- popular: 人気動画フィード
- random: ランダム動画フィード

✅ **フィルタリング機能**
- exclude_ids による動画除外
- ページネーション（limit, offset）
- 認証済み/匿名ユーザー対応

✅ **エラーハンドリング**
- 無効なパラメータの適切な処理
- データベース接続障害への対応
- レート制限の実装

### パフォーマンス要件

| メトリクス | 要件 | 測定方法 |
|-----------|------|---------|
| 平均レスポンス時間 | < 500ms | 単一リクエスト測定 |
| P95レスポンス時間 | < 300ms | 負荷テスト |
| P99レスポンス時間 | < 800ms | 負荷テスト |
| スループット | > 50 RPS | 同時接続テスト |
| エラー率 | < 5% | 負荷テスト |
| メモリ使用量増加 | < 100MB | 持続負荷テスト |

### セキュリティ要件

✅ **入力検証**
- SQLインジェクション防止
- XSSペイロード処理
- 不正なUUID形式の拒否

✅ **認証とアクセス制御**
- JWT トークン検証
- 匿名ユーザーのフォールバック
- レート制限の実装

✅ **エラー情報の適切な制御**
- 内部エラーの詳細を外部に漏らさない
- 適切なHTTPステータスコード

## テストデータ

### モックデータ構造

```javascript
// 動画データ
{
  "id": "video_1",
  "title": "Test Video 1",
  "description": "Description for test video 1",
  "thumbnail_url": "https://example.com/thumb1.jpg",
  "maker": "Maker1",
  "genre": "Action",
  "price": 1000,
  "performers": ["Performer1", "Performer2"],
  "tags": ["tag1", "tag2"]
}

// ユーザーデータ
{
  "id": "user_1",
  "email": "user1@example.com",
  "created_at": "2024-01-01T00:00:00Z"
}

// 埋め込みデータ
{
  "video_id": "video_1",
  "embedding": [0.1, 0.2, ...], // 768次元ベクター
  "updated_at": "2024-01-01T00:00:00Z"
}
```

## CI/CD統合

### GitHub Actions設定例

```yaml
name: Content API Tests

on: [push, pull_request]

jobs:
  content-api-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio httpx psutil

    - name: Run content API tests
      run: |
        cd backend/tests/integration/content
        python run_content_api_tests.py all

    - name: Upload test reports
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: test-reports
        path: backend/tests/integration/content/reports/
```

## トラブルシューティング

### 一般的な問題

1. **テスト環境の準備**
   ```bash
   # 必要なパッケージのインストール
   pip install pytest pytest-asyncio httpx psutil pytest-html pytest-cov

   # 環境変数の設定
   export SUPABASE_URL=http://localhost:54321
   export CONTENT_API_TEST_MODE=true
   ```

2. **Supabase接続エラー**
   ```bash
   # Supabaseサービスの起動確認
   supabase status

   # Edge Functionsのデプロイ確認
   supabase functions deploy content
   ```

3. **パフォーマンステストの失敗**
   - システムリソースの確認
   - 他のプロセスによる影響の排除
   - テスト用データの準備状態確認

### ログとデバッグ

```bash
# 詳細ログ付きテスト実行
pytest -v -s --log-cli-level=INFO

# 特定のテストのみデバッグ
pytest -v -s -k "test_explore_feed_basic"

# テストカバレッジの確認
pytest --cov=supabase/functions --cov-report=term-missing
```

## カスタマイズ

### 新しいテストの追加

1. 適切なテストクラスに新しいテストメソッドを追加
2. 必要に応じて新しいフィクスチャを作成
3. テストマーカーの設定（@pytest.mark.performance など）

### テスト設定のカスタマイズ

`conftest.py` でテスト設定を変更：

```python
@pytest.fixture
def custom_performance_requirements():
    return {
        'max_response_time_ms': 300,  # より厳しい要件
        'min_throughput_rps': 100
    }
```

## レポート

テスト実行後、以下のレポートが生成されます：

- **HTML レポート**: `reports/{test_name}_report.html`
- **JUnit XML**: `reports/{test_name}_results.xml`
- **JSON サマリー**: `reports/test_summary.json`

これらのレポートはCI/CDパイプラインでの結果分析や、継続的な品質監視に活用できます。