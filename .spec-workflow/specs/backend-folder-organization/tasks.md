# バックエンドフォルダ組織化 - タスク仕様書

<!-- AI Instructions: For each task, generate a _Prompt field with structured AI guidance following this format:
_Prompt: Role: [specialized developer role] | Task: [clear task description with context references] | Restrictions: [what not to do, constraints] | Success: [specific completion criteria]_
This helps provide better AI agent guidance beyond simple "work on this task" prompts. -->

## Phase 1: Python Backend モデル変換基盤構築（3時間）

- [x] 1.1 TensorFlow.js変換モジュール作成
  - File: /backend/ml_pipeline/export/keras_to_tfjs.py
  - Keras学習済みモデルをTensorFlow.js形式に変換する機能を実装
  - 既存のTwo-Towerモデル（user_tower_768.keras, item_tower_768.keras）を対象
  - Purpose: Supabase Edge FunctionsでTensorFlow.jsモデル実行を可能にする
  - _Leverage: /backend/ml_pipeline/models/rating_based_two_tower_768/_
  - _Requirements: 要件 FR-01_
  - _Prompt: Role: ML Engineer with expertise in TensorFlow model conversion and deployment | Task: Create comprehensive Keras to TensorFlow.js conversion module for Two-Tower models (user_tower_768.keras, item_tower_768.keras), ensuring 768-dimension embedding compatibility and model architecture preservation | Restrictions: Do not modify original Keras models, maintain numerical precision during conversion, ensure output compatibility with Deno/TensorFlow.js runtime | Success: Successfully converts both user and item tower models to TensorFlow.js format (.json + .bin), conversion preserves model accuracy within 1% tolerance, output models are loadable in JavaScript environment_

- [x] 1.2 モデル精度検証システム作成
  - File: /backend/ml_pipeline/export/model_validator.py
  - Keras vs TensorFlow.js推論結果比較による精度検証機能を実装
  - サンプルデータでの推論結果一致率検証
  - Purpose: 変換されたモデルの品質保証とデバッグ支援
  - _Leverage: /backend/tests/fixtures/sample_videos.json_
  - _Requirements: 要件 TR-01_
  - _Prompt: Role: QA Engineer specializing in ML model validation and numerical testing | Task: Implement comprehensive model validation system comparing Keras and TensorFlow.js inference results using sample video data, ensuring conversion accuracy and numerical stability | Restrictions: Must test with realistic data samples, do not modify model outputs during comparison, ensure statistical significance of validation tests | Success: Validation system accurately compares inference results between Keras and TensorFlow.js models, identifies any numerical discrepancies within acceptable thresholds, provides detailed accuracy reports_

- [x] 1.3 Supabase Storage統合モジュール作成
  - File: /backend/ml_pipeline/export/supabase_uploader.py
  - TensorFlow.jsモデルファイルをSupabase Storageにアップロードする機能
  - バージョン管理・無停止デプロイメント対応
  - Purpose: 変換されたモデルをSupabase Edge Functionsで利用可能にする
  - _Leverage: /backend/scripts/real_dmm_sync.js (Supabase Client patterns)_
  - _Requirements: 要件 TR-03_
  - _Prompt: Role: DevOps Engineer with expertise in cloud storage and deployment automation | Task: Create Supabase Storage integration module for uploading TensorFlow.js models with versioning and zero-downtime deployment, following existing Supabase client patterns from DMM sync scripts | Restrictions: Must not disrupt existing Edge Functions during model updates, ensure atomic uploads, maintain model version compatibility | Success: Successfully uploads TensorFlow.js models to Supabase Storage, implements proper versioning strategy, supports rollback to previous model versions, integrates with existing deployment workflows_

- [x] 1.4 モデルデプロイメント自動化スクリプト作成
  - File: /backend/scripts/model_deployment.py
  - 学習→変換→検証→デプロイメントの一連のパイプライン自動化
  - Edge Functionsモデルリロードトリガー機能
  - Purpose: モデル更新プロセスの自動化と運用効率化
  - _Leverage: /backend/ml_pipeline/training/train_768_dim_two_tower.py_
  - _Requirements: 要件 FR-04_
  - _Prompt: Role: ML Operations Engineer specializing in model deployment pipelines and automation | Task: Create comprehensive model deployment automation pipeline connecting training to production deployment, integrating existing Two-Tower training scripts with new conversion and upload modules | Restrictions: Must ensure pipeline atomicity, handle deployment failures gracefully, do not deploy models that fail validation | Success: Automated pipeline successfully trains, converts, validates and deploys Two-Tower models to production, includes rollback mechanisms, triggers Edge Functions to reload models, provides deployment status reporting_

## Phase 2: Supabase Edge Functions モデル実行基盤（3時間）

- [x] 2.1 TensorFlow.jsモデルローダー作成
  - File: /supabase/functions/_shared/model_loader.ts
  - Supabase StorageからTensorFlow.jsモデル読み込み・キャッシュ機能
  - ユーザータワー・アイテムタワーの分離実行対応
  - Purpose: Edge FunctionsでのTwo-Towerモデル実行基盤提供
  - _Leverage: /supabase/functions/_shared/types.ts_
  - _Requirements: 要件 FR-02_
  - _Prompt: Role: Frontend Developer with expertise in TensorFlow.js and Edge Functions runtime | Task: Create robust TensorFlow.js model loader for Supabase Edge Functions supporting both user and item towers with efficient caching and error handling, using existing type definitions | Restrictions: Must work within Edge Functions memory constraints, handle model loading failures gracefully, optimize for cold start performance | Success: Successfully loads TensorFlow.js models from Supabase Storage, implements efficient caching strategy, provides separate user/item tower inference, handles Edge Functions runtime limitations_

- [x] 2.2 特徴量前処理モジュール作成
  - File: /supabase/functions/_shared/feature_preprocessor.ts
  - ユーザー・アイテム特徴量をTensorFlow.js入力形式に変換
  - 現行の特徴量抽出ロジックをTypeScriptに移植
  - Purpose: 既存のPython特徴量処理とTypeScript環境での互換性確保
  - _Leverage: /supabase/functions/update_user_embedding/index.ts (extractUserFeatures)_
  - _Requirements: 要件 FR-03_
  - _Prompt: Role: Data Engineer with expertise in feature engineering and TypeScript | Task: Port existing Python feature extraction logic to TypeScript for Edge Functions compatibility, ensuring feature parity and numerical consistency with current user embedding system | Restrictions: Must maintain exact feature compatibility with existing Python implementation, ensure numerical precision, do not alter feature engineering logic | Success: Feature preprocessing produces identical results to Python implementation, supports both user and item feature extraction, optimized for Edge Functions performance_

- [ ] 2.3 学習済みTwo-Tower推論API作成
  - File: /supabase/functions/user_embedding_v2/index.ts
  - TensorFlow.js Two-Towerモデルによるリアルタイムユーザーエンベディング生成
  - 既存の簡易線形モデルからのアップグレード版
  - Purpose: 高精度なユーザーエンベディングによる推薦精度向上
  - _Leverage: /supabase/functions/update_user_embedding/index.ts_
  - _Requirements: 要件 FR-02_
  - _Prompt: Role: Full-stack Developer with expertise in machine learning APIs and Edge Functions | Task: Implement high-accuracy user embedding API using trained Two-Tower model, upgrading from existing simple linear model while maintaining API compatibility and performance | Restrictions: Must maintain backward compatibility with existing API contracts, handle model loading errors with fallback to simple model, ensure sub-second response times | Success: API generates high-accuracy user embeddings using TensorFlow.js Two-Tower model, maintains API compatibility, includes proper error handling and fallback mechanisms, meets performance requirements_

- [ ] 2.4 アイテムエンベディング生成API作成
  - File: /supabase/functions/item_embedding/index.ts
  - Two-Towerアイテムタワーによるコンテンツエンベディング生成
  - DMM APIデータからの特徴量抽出・エンベディング生成
  - Purpose: コンテンツの高精度ベクトル表現生成
  - _Leverage: /backend/data-processing/utils/batch_data_integrator.py (feature patterns)_
  - _Requirements: 要件 FR-02_
  - _Prompt: Role: Backend Developer specializing in content processing and embedding generation | Task: Create item embedding API using Two-Tower item tower for content vectorization, extracting features from DMM API data and generating high-quality item embeddings | Restrictions: Must handle various content metadata formats, ensure embedding consistency, optimize for batch processing of multiple items | Success: Successfully generates item embeddings from DMM content data, supports both single and batch processing modes, maintains embedding quality and consistency, integrates with video_embeddings table_

## Phase 3: DMM統合強化とコンテンツ処理（2時間）

- [ ] 3.1 DMM統合エンベディング生成機能強化
  - File: /supabase/functions/dmm_content_sync/index.ts
  - 既存DMM同期機能にアイテムエンベディング生成を統合
  - 大量コンテンツの並列エンベディング処理対応
  - Purpose: DMM APIデータ取得と同時に高精度エンベディング生成
  - _Leverage: /supabase/functions/dmm_sync/index.ts_
  - _Requirements: 要件 FR-04_
  - _Prompt: Role: Integration Engineer with expertise in API integration and parallel processing | Task: Enhance existing DMM sync functionality to include item embedding generation using Two-Tower model, supporting large-scale content processing with parallel execution | Restrictions: Must maintain existing DMM API rate limits, handle processing failures gracefully, ensure data consistency between videos and embeddings tables | Success: DMM sync successfully generates embeddings for all synced content, processes large batches efficiently, maintains data integrity, handles API rate limits and errors appropriately_

- [ ] 3.2 バッチコンテンツ処理システム作成
  - File: /supabase/functions/batch_content_processing/index.ts
  - 既存コンテンツの一括エンベディング再生成機能
  - 進捗追跡・エラーハンドリング・再試行機能
  - Purpose: 既存1000件のDMMビデオデータを新モデルでエンベディング更新
  - _Leverage: /backend/scripts/analyze_dmm_data.js (data analysis patterns)_
  - _Requirements: 要件 NFR-01_
  - _Prompt: Role: Backend Developer specializing in batch processing and data migration | Task: Create robust batch processing system for re-generating embeddings for existing 1000 DMM videos using new Two-Tower model, with progress tracking and error recovery | Restrictions: Must not disrupt live system, handle processing interruptions gracefully, maintain processing state for resumability | Success: Successfully processes all existing videos for embedding regeneration, provides accurate progress tracking, handles errors and retries appropriately, minimal impact on live system performance_

- [ ] 3.3 コンテンツ品質管理システム作成
  - File: /supabase/functions/content_quality_monitor/index.ts
  - エンベディング品質メトリクス計算・監視機能
  - 異常検知・アラート機能
  - Purpose: コンテンツエンベディングの品質保証と運用監視
  - _Leverage: /backend/tests/performance/test_ml_pipeline.py (quality metrics patterns)_
  - _Requirements: 要件 NFR-02_
  - _Prompt: Role: ML Operations Engineer with expertise in model monitoring and quality assurance | Task: Implement content quality monitoring system for embedding generation, including quality metrics calculation and anomaly detection based on existing performance testing patterns | Restrictions: Must not impact production performance, ensure monitoring accuracy, provide actionable alerts only | Success: Quality monitoring accurately tracks embedding generation performance, detects anomalies in embedding quality, provides meaningful alerts for operational issues_

## Phase 4: フォルダ構造統合と整理（1時間）

- [ ] 4.1 Python Backend フォルダ統合実行
  - Files: Multiple (backend reorganization)
  - `/backend/edge_functions/` → `/backend/shared/` 名前変更
  - ルートディレクトリ重複フォルダの`/backend/`への統合
  - Purpose: 統一されたPython Backend構造の実現
  - _Leverage: existing folder structure analysis_
  - _Requirements: 要件 BR-01, BR-02_
  - _Prompt: Role: DevOps Engineer with expertise in codebase reorganization and file system operations | Task: Execute systematic folder reorganization consolidating all Python backend code into /backend/ directory, renaming edge_functions to shared, and eliminating duplicate folders | Restrictions: Must preserve all file contents, maintain git history where possible, ensure atomic operations to prevent partial completion states | Success: All Python backend code consolidated under /backend/, no duplicate folders between root and backend directories, folder structure matches design specification_

- [ ] 4.2 インポート・参照パス一括更新
  - Files: Multiple Python files (import statements)
  - 移動されたモジュールのインポートパス更新
  - 設定ファイル・スクリプト内パス参照更新
  - Purpose: フォルダ移動後の動作保証
  - _Leverage: existing Python module structure_
  - _Requirements: 要件 FR-03_
  - _Prompt: Role: Python Developer with expertise in module management and refactoring | Task: Update all import statements and path references affected by folder reorganization, ensuring all Python modules resolve correctly after directory restructuring | Restrictions: Must maintain exact functionality, do not alter module behavior, ensure all imports resolve correctly | Success: All Python imports resolve correctly after folder reorganization, no broken module references, existing functionality preserved exactly_

- [ ] 4.3 設定ファイル・ドキュメント更新
  - Files: pyproject.toml, README.md, CLAUDE.md, documentation
  - フォルダ構造変更に対応した設定・ドキュメント更新
  - アーキテクチャドキュメントの更新
  - Purpose: 変更されたプロジェクト構造の正確な反映
  - _Leverage: /home/devel/dev/adult_matching/CLAUDE.md_
  - _Requirements: 要件 DR-01_
  - _Prompt: Role: Technical Writer with expertise in software documentation and project configuration | Task: Update all configuration files and documentation to reflect new folder structure, ensuring accuracy and consistency across project documentation | Restrictions: Must maintain documentation accuracy, do not alter functional configuration unnecessarily, ensure consistency with actual implementation | Success: All documentation and configuration accurately reflects new folder structure, development guides are updated and functional, project setup instructions are correct_

## Phase 5: 包括的テスト・検証（2時間）

- [ ] 5.1 モデル変換・デプロイメント統合テスト
  - File: /backend/tests/integration/test_model_deployment.py
  - 学習→変換→デプロイメントの完全パイプラインテスト
  - TensorFlow.js変換精度検証
  - Purpose: モデルデプロイメントプロセスの品質保証
  - _Leverage: /backend/tests/conftest.py, /backend/tests/fixtures/_
  - _Requirements: 全要件の統合検証_
  - _Prompt: Role: QA Engineer specializing in ML pipeline testing and integration verification | Task: Create comprehensive integration tests for complete model deployment pipeline from training to production deployment, verifying conversion accuracy and deployment success | Restrictions: Must test realistic scenarios, ensure tests are deterministic, do not impact production systems during testing | Success: Integration tests verify complete model pipeline functionality, conversion accuracy within acceptable thresholds, deployment process robust and reliable_

- [ ] 5.2 Supabase Edge Functions エンドツーエンドテスト
  - File: /backend/tests/integration/test_supabase_embedding_api.py
  - ユーザー・アイテムエンベディング生成APIの統合テスト
  - DMM統合・バッチ処理機能のテスト
  - Purpose: Supabase Edge Functions全機能の動作保証
  - _Leverage: /backend/tests/integration/test_edge_functions.py_
  - _Requirements: 全要件の統合検証_
  - _Prompt: Role: Full-stack QA Engineer with expertise in API testing and Edge Functions | Task: Create comprehensive end-to-end tests for all Supabase Edge Functions including user/item embedding generation and DMM integration, extending existing Edge Functions tests | Restrictions: Must test real API endpoints, handle Edge Functions cold starts, ensure test isolation and cleanup | Success: E2E tests verify all Edge Functions work correctly with TensorFlow.js models, API responses are accurate and performant, DMM integration functions properly_

- [ ] 5.3 パフォーマンス・品質検証テスト
  - File: /backend/tests/performance/test_embedding_performance.py
  - エンベディング生成速度・精度の定量的評価
  - 簡易線形モデル vs Two-Tower モデル比較
  - Purpose: 新システムのパフォーマンス・品質定量評価
  - _Leverage: /backend/tests/performance/test_ml_pipeline.py_
  - _Requirements: 要件 NFR-01, NFR-02_
  - _Prompt: Role: Performance Engineer with expertise in ML system benchmarking and quality metrics | Task: Implement comprehensive performance and quality testing for embedding generation, comparing simple linear model vs Two-Tower model performance and accuracy | Restrictions: Must provide statistically significant results, ensure fair comparison conditions, measure realistic production scenarios | Success: Performance tests provide clear metrics on embedding generation speed and quality, comparison between models is objective and comprehensive, results demonstrate system improvements_

- [ ] 5.4 本番デプロイメント・切り替え検証
  - File: /backend/scripts/production_deployment_verification.py
  - 本番環境でのモデル切り替え・フォールバック機能テスト
  - 既存システムとの互換性確認
  - Purpose: 本番環境での安全なシステム切り替え保証
  - _Leverage: existing production deployment patterns_
  - _Requirements: 要件 BR-04_
  - _Prompt: Role: Site Reliability Engineer with expertise in production deployments and system reliability | Task: Verify production deployment readiness including model switching, fallback mechanisms, and backward compatibility with existing system components | Restrictions: Must not disrupt production service, ensure rollback capabilities, test real production scenarios safely | Success: Production deployment verified as safe and reliable, fallback mechanisms work correctly, system maintains backward compatibility, zero-downtime deployment achieved_

## 包括的テストと検証要件

### テスト実装標準
- **場所**: `tests/` ディレクトリ配下に適切な分類
- **フレームワーク**: pytest (Python), Jest/Deno Test (TypeScript)
- **実行**: 自動実行機能対応
- **レポート**: 明確な合格/不合格基準と詳細レポート
- **CI/CD**: 継続的インテグレーションパイプライン統合

### 最終検証基準
- **機能検証**: 全API エンドポイントが正常動作
- **パフォーマンス検証**: エンベディング生成速度 <100ms (user), <1s (batch)
- **品質検証**: Two-Tower vs 簡易モデル推薦精度向上確認
- **信頼性検証**: エラーハンドリング・フォールバック機能
- **セキュリティ検証**: データ保護・アクセス制御

### エンフォースメント・ポリシー
- **テストなしの本番デプロイなし**: 全機能に対応するテストが必要
- **全実装にテストスイート対応**: テストのない実装は不完全
- **テストギャップは即座に仕様更新**: テスト不備は仕様更新が必要
- **品質ゲート**: テスト検証なしでの本番デプロイ防止