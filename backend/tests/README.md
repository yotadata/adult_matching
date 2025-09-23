# バックエンドテストスイート

バックエンドリファクタリングの包括的なテスト環境

## 📁 ディレクトリ構成

```
tests/
├── conftest.py              # pytest設定とフィクスチャ
├── pytest.ini              # pytest設定ファイル
├── README.md               # このファイル
├── integration/            # 統合テスト
│   ├── test_edge_functions.py    # Edge Functions統合テスト
│   └── test_api_endpoints.py     # APIエンドポイントテスト
├── unit/                   # ユニットテスト
│   ├── test_shared_utils.py      # 共通ユーティリティテスト
│   ├── test_database.py          # データベース操作テスト
│   └── test_auth.py              # 認証機能テスト
├── performance/            # パフォーマンステスト
│   ├── test_ml_pipeline.py       # MLパイプライン性能テスト
│   └── test_load.py              # 負荷テスト
└── fixtures/               # テストデータフィクスチャ
    ├── sample_videos.json        # サンプル動画データ
    └── sample_embeddings.npy     # サンプルエンベディング
```

## 🧪 テスト分類

### ユニットテスト (unit/)
- **目的**: 個々の関数・クラスの動作確認
- **特徴**: 高速実行、外部依存なし、モック使用
- **実行時間**: < 1秒/テスト

### 統合テスト (integration/)
- **目的**: コンポーネント間の連携確認
- **特徴**: データベース・API接続、実際のデータフロー
- **実行時間**: 1-30秒/テスト

### パフォーマンステスト (performance/)
- **目的**: 性能要件の確認、ボトルネック特定
- **特徴**: 時間・メモリ・CPU使用量測定
- **実行時間**: 10秒-数分/テスト

## 🚀 実行方法

### 基本実行
```bash
# 全テスト実行
pytest

# 特定カテゴリのテスト実行
pytest tests/unit/           # ユニットテストのみ
pytest tests/integration/   # 統合テストのみ
pytest tests/performance/   # パフォーマンステストのみ
```

### マーカーを使用した実行
```bash
# ユニットテストのみ実行
pytest -m unit

# 統合テストのみ実行
pytest -m integration

# パフォーマンステストのみ実行
pytest -m performance

# 遅いテストを除外
pytest -m "not slow"

# データベーステストのみ実行
pytest -m database
```

### 詳細オプション
```bash
# 詳細出力
pytest -v

# カバレッジレポート付き
pytest --cov=edge_functions --cov-report=html

# 並列実行
pytest -n auto

# 特定のテストファイル実行
pytest tests/unit/test_shared_utils.py

# 特定のテスト関数実行
pytest tests/unit/test_shared_utils.py::TestDatabaseUtils::test_execute_query_success
```

## 🔧 環境設定

### 必要な環境変数
```bash
# テスト実行に必要な環境変数
export TESTING=true
export DATABASE_URL=postgresql://test:test@localhost:5432/test_db
export SUPABASE_URL=https://test.supabase.co
export SUPABASE_ANON_KEY=test_anon_key
export DMM_API_ID=test_api_id
export DMM_AFFILIATE_ID=test_affiliate_id

# オプション: 統合テスト有効化
export RUN_INTEGRATION_TESTS=1

# オプション: パフォーマンステスト有効化
export RUN_PERFORMANCE_TESTS=1

# オプション: 遅いテストスキップ
export SKIP_SLOW_TESTS=1
```

### テスト用データベース準備
```bash
# PostgreSQLテストデータベース作成
createdb test_db

# スキーマ適用
psql test_db < ../db/migrations/20250818000000000_initial.sql
```

## 📊 性能要件

### レスポンス時間要件
- **Edge Functions**: < 2秒
- **データベースクエリ**: < 1秒
- **エンベディング生成**: < 10秒/100件
- **モデル推論**: < 100ms/件

### メモリ使用量要件
- **Edge Functions**: < 128MB
- **MLパイプライン**: < 1GB
- **データ処理**: < 2GB

### スループット要件
- **API呼び出し**: > 10 req/sec
- **エンベディング生成**: > 10 items/sec
- **データベース操作**: > 100 ops/sec

## 🧩 テストフィクスチャ

### 共通フィクスチャ (conftest.py)
- `mock_supabase_client`: Supabaseクライアントのモック
- `mock_database_connector`: データベースコネクターのモック
- `sample_video_data`: サンプル動画データ
- `sample_user_data`: サンプルユーザーデータ
- `sample_embeddings`: サンプルエンベディングデータ
- `temp_dir`: 一時ディレクトリ
- `mock_config`: 設定オブジェクトのモック

### フィクスチャ使用例
```python
def test_video_processing(sample_video_data, mock_database_connector):
    """フィクスチャを使用したテスト例"""
    # サンプルデータを使用
    videos = sample_video_data
    
    # モックデータベースを使用
    result = mock_database_connector.save_scraped_videos(videos)
    
    assert result >= 0
```

## 🔍 テストカバレッジ

### カバレッジ目標
- **全体**: ≥ 80%
- **共通ユーティリティ**: ≥ 95%
- **Edge Functions**: ≥ 85%
- **MLパイプライン**: ≥ 80%
- **データ処理**: ≥ 90%

### カバレッジレポート確認
```bash
# HTMLレポート生成
pytest --cov=edge_functions --cov-report=html

# レポート確認
open htmlcov/index.html
```

## 🚨 継続的インテグレーション

### GitHub Actionsワークフロー例
```yaml
name: Backend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install -r backend/requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test_db
        RUN_INTEGRATION_TESTS: 1
      run: |
        pytest tests/ --cov=edge_functions --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🛠️ トラブルシューティング

### よくある問題

#### 1. データベース接続エラー
```
psycopg2.OperationalError: could not connect to server
```
**解決方法**:
- PostgreSQLが起動していることを確認
- 接続文字列が正しいことを確認
- テスト用データベースが存在することを確認

#### 2. インポートエラー
```
ModuleNotFoundError: No module named 'edge_functions'
```
**解決方法**:
- `conftest.py`でパス設定を確認
- `__init__.py`ファイルが存在することを確認

#### 3. 非同期テストエラー
```
RuntimeError: There is no current event loop
```
**解決方法**:
- `@pytest.mark.asyncio`デコレータを使用
- `conftest.py`のイベントループフィクスチャを確認

#### 4. パフォーマンステスト失敗
```
AssertionError: 処理時間が長すぎます: 15.2秒
```
**解決方法**:
- システムリソースを確認
- テストデータサイズを調整
- 性能要件を環境に応じて調整

## 📈 テスト結果の分析

### メトリクス収集
- テスト実行時間
- カバレッジ率
- 成功/失敗率
- パフォーマンス指標

### レポート生成
```bash
# JUnit形式レポート
pytest --junitxml=test-results.xml

# HTMLレポート
pytest --html=test-report.html --self-contained-html
```

## 🎯 ベストプラクティス

1. **テスト命名**: `test_<機能>_<条件>_<期待結果>`
2. **フィクスチャ使用**: 共通データは必ずフィクスチャ化
3. **モック適用**: 外部依存は積極的にモック化
4. **アサーション**: 明確で具体的なアサーション
5. **テストデータ**: 最小限かつ意味のあるデータ
6. **クリーンアップ**: テスト後の状態初期化

## 🔄 メンテナンス

### 定期的な作業
- テストデータの更新
- パフォーマンス要件の見直し
- カバレッジ率の改善
- 不安定なテストの修正

### 新機能追加時
- 対応するテストの追加
- フィクスチャの更新
- ドキュメントの更新