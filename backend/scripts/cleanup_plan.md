# Script Cleanup and Archival Plan

## 移行完了したスクリプト

### ✅ 新しい場所に移行済み
- `train_production_two_tower.py` → `backend/ml/training/scripts/`
- `train_768_dim_two_tower.py` → `backend/ml/training/scripts/`
- `verify_768_model.py` → `backend/ml/training/scripts/`
- `standardize_models_768.py` → `backend/ml/scripts/` (既存)
- `real_dmm_sync.js` → `backend/data/ingestion/scripts/dmm_api_sync.js`
- `analyze_dmm_data.js` → `backend/data/analysis/scripts/`
- `analyze_review_dates.js` → `backend/data/analysis/scripts/`

## 重複・統合が必要なスクリプト

### ML Training Scripts (要統合)
- `train_two_tower_comprehensive.py` - 包括的だが新アーキテクチャと不整合
- `train_two_tower_model.py` - 基本版、新版に機能統合済み
- `simple_two_tower_test.py` - テスト用、テストスイートに統合予定
- `test_training_components.py` - コンポーネントテスト、新テストに統合
- `test_768_pgvector_integration.py` - 統合テスト、新テストスイートで実装

### Data Sync Scripts (要整理)
- `efficient_dmm_bulk_sync.js` - 大容量同期用、メインに機能統合可能
- `mega_dmm_sync_200k.js` - 大規模同期、パフォーマンス改善版として保持
- `multi_sort_dmm_sync.js` - マルチソート機能、メインに統合
- `test_dmm_sync_small.js` - テスト用、テストスイートに移行

### Data Processing Scripts (要整理)  
- `accurate_content_linking.js` - コンテンツリンキング改善版
- `content_id_linking.js` - 基本版、上記に統合
- `diagnose_database_issue.js` - 診断ツール、メンテナンス用として保持

### Legacy/Experimental (アーカイブ・削除候補)
- `ingestFanzaData.ts` - `fanza_ingest.ts`と重複
- `sync-dmm.ts` - TypeScript版だが使用実績少ない

## クリーンアップアクション

### Phase 1: Safe Migration (実行済み)
- [x] 本番使用中スクリプトの新場所への安全な移行
- [x] 新アーキテクチャでの動作確認
- [x] ScriptManagerによる統合管理

### Phase 2: Archive Legacy Scripts 
```bash
# レガシースクリプトをarchiveディレクトリに移動
mkdir -p backend/scripts/archive
mv backend/scripts/train_two_tower_comprehensive.py backend/scripts/archive/
mv backend/scripts/train_two_tower_model.py backend/scripts/archive/ 
mv backend/scripts/simple_two_tower_test.py backend/scripts/archive/
mv backend/scripts/test_training_components.py backend/scripts/archive/
mv backend/scripts/ingestFanzaData.ts backend/scripts/archive/
```

### Phase 3: Consolidate Similar Scripts
1. **DMM Sync統合**
   - `real_dmm_sync.js`をベースに他の機能を統合
   - バッチサイズ・パフォーマンス最適化機能を追加
   
2. **Content Linking統合** 
   - `accurate_content_linking.js`をメインに
   - `content_id_linking.js`の機能をマージ

### Phase 4: Final Cleanup
- アーカイブされたスクリプトの動作確認
- 30日後に実際の削除実行
- ドキュメント更新

## 保持すべき重要スクリプト

### 本番運用スクリプト
- `dmm_api_sync.js` (旧 real_dmm_sync.js) - 現在本番稼働中
- `analyze_dmm_data.js` - データ品質監視で使用
- `train_production_two_tower.py` - 本番モデルトレーニング

### 開発・テストツール  
- `verify_768_model.py` - モデル検証で必須
- `diagnose_database_issue.js` - 障害対応で有用
- `mega_dmm_sync_200k.js` - 大規模データ同期で必要

### 特殊用途
- `analyze_review_dates.js` - データ分析レポート用
- `fanza_ingest.ts` - FANZA専用取り込み
- `model_deployment.py` - デプロイメント自動化

## Script Manager統合

新しい統合管理により：
- すべてのスクリプトが一元管理される
- 実行履歴とログが統合される  
- パイプライン実行が自動化される
- 依存関係とエラー処理が改善される

使用例:
```bash
# スクリプト一覧
python backend/scripts/script_manager.py --list

# MLパイプライン実行
python backend/scripts/script_manager.py --ml-pipeline

# データ同期パイプライン実行  
python backend/scripts/script_manager.py --data-pipeline

# 実行履歴確認
python backend/scripts/script_manager.py --history
```