# Script Inventory and Migration Plan

## 目的
プロジェクト内の全スクリプトを監査・分類し、新しいbackend/ml、backend/dataアーキテクチャに統合する

## 発見されたスクリプト分類

### 1. ML Training Scripts (Python)
**場所**: `backend/scripts/`
- `simple_two_tower_test.py` - 簡単なTwo-Towerテスト
- `train_two_tower_comprehensive.py` - 包括的Two-Towerトレーニング  
- `test_768_pgvector_integration.py` - 768次元PgVector統合テスト
- `train_two_tower_model.py` - Two-Towerモデルトレーニング
- `model_deployment.py` - モデルデプロイメント
- `train_production_two_tower.py` - 本番Two-Towerトレーニング
- `test_training_components.py` - トレーニングコンポーネントテスト
- `train_768_dim_two_tower.py` - 768次元Two-Towerトレーニング
- `verify_768_model.py` - 768次元モデル検証

### 2. Data API Synchronization Scripts (Node.js/TypeScript)
**場所**: `backend/scripts/`
- `real_dmm_sync.js` - DMM APIリアル同期（メインスクリプト）
- `efficient_dmm_bulk_sync.js` - 効率的DMM一括同期
- `test_dmm_sync_small.js` - DMM同期小規模テスト
- `mega_dmm_sync_200k.js` - 大規模DMM同期（20万件）
- `multi_sort_dmm_sync.js` - マルチソートDMM同期
- `fanza_ingest.ts` - FANZA データ取り込み
- `ingestFanzaData.ts` - FANZA データ取り込み（重複？）
- `sync-dmm.ts` - DMM同期TypeScriptバージョン

### 3. Data Analysis Scripts (Node.js)
**場所**: `backend/scripts/`  
- `analyze_dmm_data.js` - DMM データ品質分析
- `analyze_review_dates.js` - レビュー日付分析
- `accurate_content_linking.js` - 正確なコンテンツリンキング
- `content_id_linking.js` - コンテンツIDリンキング
- `diagnose_database_issue.js` - データベース問題診断

### 4. ML Pipeline Components (Python) - Already Migrated
**新しい場所**: `backend/ml/`
- `standardize_models_768.py` - 768次元モデル標準化（既に移行済み）
- `train_standard_model.py` - 標準モデルトレーニング（既に移行済み）

### 5. Legacy/Redundant Scripts 
**要確認・削除候補**:
- 重複したトレーニングスクリプト
- 古いバージョンのAPIスクリプト
- 試験的な実装ファイル

## 移行計画

### Phase 1: Script Categorization and Organization
1. **ML Training Scripts** → `backend/ml/training/scripts/`
2. **Data Processing Scripts** → `backend/data/scripts/`
3. **API Sync Scripts** → `backend/data/ingestion/scripts/`
4. **Analysis Scripts** → `backend/data/analysis/scripts/`

### Phase 2: Script Modernization
1. スクリプト統合・重複除去
2. 新しいアーキテクチャとの統合
3. 設定管理の統一化
4. エラーハンドリングの改善

### Phase 3: Documentation and Testing
1. 各スクリプトの詳細ドキュメント作成
2. 使用例とベストプラクティス
3. テストケース作成
4. CI/CD統合

## 推奨アクション

### 即座に移行すべきスクリプト:
- `real_dmm_sync.js` - 本番で使用中、データ取り込みシステムに統合
- `analyze_dmm_data.js` - データ品質管理システムに統合
- ML トレーニングスクリプト群 - backend/mlに統合

### 重複除去・統合が必要:
- 複数のTwo-Towerトレーニングスクリプト
- 複数のDMM同期スクリプト  
- FANZAデータ取り込みスクリプト

### 削除・アーカイブ候補:
- 実験的・テスト用スクリプト（目的達成済み）
- 古いバージョンの重複スクリプト
- 使用されていない診断スクリプト

## 次のステップ
1. カテゴリ別スクリプト移行実行
2. 統合・モダナイゼーション
3. テスト・検証
4. ドキュメント更新