# End-to-End System Tests

エンドツーエンドシステムテスト - 包括的なユーザーワークフローとパフォーマンステスト

## 概要

このE2Eテストスイートは、Adult Matching Backendの完全なシステム機能をテストし、以下の要件を検証します：

- **ユーザーワークフロー**: 新規ユーザーから ヘビーユーザーまでの全ジャーニー
- **パフォーマンス要件**: 推薦<500ms、50+ RPS、並行ユーザー対応
- **システム信頼性**: 障害耐性、エラーハンドリング、グレースフルデグラデーション

## テスト構成

### 1. ユーザーワークフローテスト (`workflows/`)

#### `test_user_workflows.py`
- **新規ユーザーオンボーディング**: 初回利用から学習後推薦まで
- **アクティブユーザーブラウジング**: 複数ページセッション
- **ヘビーユーザーセッション**: 高頻度インタラクション
- **混合ユーザーシナリオ**: 並行ユーザーセッション

**検証項目:**
- ワークフロー完了率 ≥ 95%
- セッション継続性
- ユーザーエクスペリエンス品質
- レスポンス時間一貫性

### 2. パフォーマンス要件テスト (`performance/`)

#### `test_performance_requirements.py`
- **推薦レスポンス時間**: <500ms要件検証
- **並行ユーザースケーラビリティ**: 1-50ユーザー負荷テスト
- **メモリ使用量**: 負荷下でのメモリ効率性
- **継続負荷安定性**: 60秒継続テスト

**検証項目:**
- 平均レスポンス時間 < 500ms
- P95レスポンス時間 < 750ms
- スループット ≥ 50 RPS
- メモリ増加 < 200MB
- エラー率 ≤ 5%

### 3. システム信頼性テスト (`system/`)

#### `test_system_reliability.py`
- **エラーハンドリング**: 各種障害シナリオ対応
- **データ一貫性**: 障害下でのデータ整合性
- **グレースフルデグラデーション**: 性能劣化時の対応
- **リソース枯渇対応**: メモリ/CPU/ディスク制限
- **並行障害処理**: 複数障害同時発生

**検証項目:**
- 障害回復率 ≥ 30%
- データ一貫性率 ≥ 95%
- デグラデーション許容性 ≥ 75%
- 並行障害耐性 ≥ 60%

## テスト実行

### 完全E2Eテスト実行
```bash
# 全テストスイート実行
python tests/e2e/test_e2e_runner.py

# 特定スイート実行
python -m pytest tests/e2e/workflows/ -v
python -m pytest tests/e2e/performance/ -v
python -m pytest tests/e2e/system/ -v
```

### 設定とフィクスチャ

#### `conftest.py`
- **システム設定**: パフォーマンス閾値、テストデータサイズ
- **パフォーマンス監視**: リアルタイムメトリクス収集
- **テストユーザープロファイル**: 新規・アクティブ・ヘビーユーザー
- **ビデオカタログ**: 100件のテストビデオデータ
- **モックサービス**: ML推薦、データベース、システム統合

#### `test_e2e_runner.py`
- **テスト統合実行**: 全スイートの自動実行
- **レポート生成**: JSON形式の詳細レポート
- **システムヘルスチェック**: メモリ、CPU、ディスク、ネットワーク
- **要件検証**: パフォーマンス・信頼性要件の自動判定

## パフォーマンス監視

### 監視メトリクス
- **レスポンス時間**: 平均、P95、P99、最大値
- **スループット**: 操作/秒、リクエスト/秒
- **成功率**: 操作成功率、エラー率
- **リソース使用量**: メモリ、CPU、ディスク

### 自動測定
```python
# パフォーマンス監視例
performance_monitor.start_monitoring("test_name")
# ... 操作実行
performance_monitor.record_operation("test_name", "operation", latency_ms, success)
performance_monitor.stop_monitoring("test_name")
summary = performance_monitor.get_summary("test_name")
```

## エラーハンドリングテスト

### 障害シナリオ
1. **ネットワーク障害**: タイムアウト、接続エラー
2. **データベース障害**: 接続失敗、高レイテンシ
3. **ML サービス障害**: 推薦エンジン停止、遅延
4. **リソース制限**: メモリ不足、CPU過負荷
5. **レート制限**: API制限、同時接続制限

### 回復機能検証
- **フォールバック**: 代替システムへの切り替え
- **キャッシュ活用**: 障害時の過去データ利用
- **部分機能**: 主要機能の継続提供
- **自動回復**: 障害解決後の自動復旧

## レポートと分析

### 自動生成レポート
- **テスト実行サマリー**: 全体成功率、実行時間
- **スイート別結果**: ワークフロー、パフォーマンス、信頼性
- **パフォーマンス分析**: 閾値との比較、傾向分析
- **システムヘルス**: リソース使用状況、警告
- **要件適合性**: 仕様要件との適合度

### レポート出力先
```
test_reports/
├── e2e_test_report_YYYYMMDD_HHMMSS.json
└── performance_metrics/
    ├── latency_analysis.json
    ├── throughput_analysis.json
    └── reliability_metrics.json
```

## CI/CD統合

### 継続的テスト実行
```yaml
# .github/workflows/e2e-tests.yml (例)
name: E2E Tests
on: [push, pull_request]
jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run E2E Tests
        run: python tests/e2e/test_e2e_runner.py
      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: e2e-test-reports
          path: test_reports/
```

### 品質ゲート
- **パフォーマンス要件**: 全て満たさない場合はデプロイ停止
- **信頼性要件**: クリティカル障害対応必須
- **ワークフロー完了率**: 95%以上維持

## カスタマイズ

### 設定変更
```python
# conftest.py での設定カスタマイズ
system_config = {
    'performance_thresholds': {
        'recommendation_latency_ms': 300,  # より厳しい要件
        'throughput_rps': 100,             # より高いスループット
        'concurrent_users': 200            # より多くの並行ユーザー
    }
}
```

### 新しいテストシナリオ追加
```python
# 新しいワークフローテスト追加例
@pytest.mark.e2e
@pytest.mark.workflow
async def test_premium_user_workflow(system_under_test, performance_monitor):
    # プレミアムユーザー向け特別ワークフロー
    pass
```

## トラブルシューティング

### よくある問題
1. **テストタイムアウト**: `timeout`設定の調整
2. **メモリ不足**: `max_memory_mb`制限の緩和
3. **並行実行エラー**: `concurrent_users`数の削減
4. **モック設定エラー**: フィクスチャの依存関係確認

### ログレベル調整
```bash
# デバッグモードでの実行
LOG_LEVEL=DEBUG python tests/e2e/test_e2e_runner.py
```

## 要件対応マトリックス

| 要件 | テストカバレッジ | 検証方法 |
|------|------------------|----------|
| 推薦レスポンス<500ms | ✅ | `test_recommendation_latency_requirement` |
| 50+ RPS対応 | ✅ | `test_concurrent_user_scalability` |
| 並行ユーザー対応 | ✅ | `test_sustained_load_stability` |
| エラーハンドリング | ✅ | `test_error_handling_and_recovery` |
| データ一貫性 | ✅ | `test_data_consistency_under_failure` |
| グレースフルデグラデーション | ✅ | `test_graceful_degradation` |
| ユーザーワークフロー | ✅ | `test_new_user_onboarding_workflow` |
| システム安定性 | ✅ | `test_concurrent_failure_scenarios` |

## 成功基準

### クリティカル要件
- [ ] 推薦レスポンス時間 < 500ms (平均)
- [ ] システム可用性 > 99%
- [ ] データ一貫性 > 95%
- [ ] ワークフロー完了率 > 95%

### パフォーマンス要件
- [ ] スループット > 50 RPS
- [ ] 並行ユーザー 50+ 対応
- [ ] メモリ効率性 (増加 < 200MB)
- [ ] CPU効率性 (< 80% 継続使用)

### 信頼性要件
- [ ] 障害回復機能
- [ ] エラーハンドリング包括性
- [ ] リソース枯渇対応
- [ ] 複数障害同時対応

このE2Eテストスイートにより、Adult Matching Backendの完全なシステム品質が保証され、本番環境での信頼性の高い運用が可能になります。