# タスク文書

<!-- AI Instructions: For each task, generate a _Prompt field with structured AI guidance following this format:
_Prompt: Role: [specialized developer role] | Task: [clear task description with context references] | Restrictions: [what not to do, constraints] | Success: [specific completion criteria]_
This helps provide better AI agent guidance beyond simple "work on this task" prompts. -->

## フェーズ1: 基本実装・検証

### 特徴エンジニアリング改良

- [x] 1. リアルユーザー特徴抽出器の実装
  - File: ml_pipeline/preprocessing/real_user_feature_extractor.py
  - user_video_decisionsテーブルからの行動パターン抽出
  - 疑似ユーザーデータからリアルデータへの移行対応
  - Purpose: 現在の評価ベースから実際のユーザー行動ベースへの特徴抽出進化
  - _Leverage: 既存のRatingBasedTwoTowerTrainerの特徴処理パターン_
  - _Requirements: 要件2_
  - _Prompt: Role: データサイエンティスト、特徴エンジニアリング専門家 | Task: 要件2に基づいてuser_video_decisionsテーブル（like/nope）から効率的にユーザー行動特徴を抽出し、既存の評価ベース疑似ユーザーデータとのスムーズな移行を実現 | Restrictions: 既存の特徴処理パイプライン互換性維持、768次元出力の保持、疑似データとの混在対応 | Success: リアルユーザーデータから意味のある行動特徴が抽出され、推薦精度が現在の99.7% AUC-PR以上を維持_

- [x] 2. 強化されたアイテム特徴処理の実装
  - File: ml_pipeline/preprocessing/enhanced_item_feature_processor.py
  - 現在の32,304アイテムに対する最適化された特徴抽出
  - TF-IDFとカテゴリ特徴の次元調整
  - Purpose: ビデオメタデータからより表現力の高い特徴量生成
  - _Leverage: 既存のtext_vectorizer、genre_encoder、maker_encoderパターン_
  - _Requirements: 要件2_
  - _Prompt: Role: NLP エンジニア、テキスト処理専門家 | Task: 要件2に従って現在の32,304ビデオデータから高品質な特徴を抽出し、TF-IDFベクトル（1000次元）とカテゴリエンコーディングの最適化を実現 | Restrictions: 既存エンコーダとの互換性維持、メモリ使用量の制限、日本語コンテンツ対応 | Success: アイテム特徴の表現力向上により、類似ビデオ検索の精度が向上し、推薦多様性が適切に保たれる_

### Two-Towerアーキテクチャ改良

- [x] 3. 768次元対応Two-Towerモデルの構築
  - File: ml_pipeline/training/enhanced_two_tower_trainer.py
  - 現在の64次元から768次元への拡張
  - ユーザー・アイテムタワーの対称設計
  - Purpose: データベース埋め込みテーブルとの完全な互換性確保
  - _Leverage: 既存のTwoTowerTrainer、RatingBasedTwoTowerTrainerアーキテクチャ_
  - _Requirements: 要件1_
  - _Prompt: Role: 深層学習エンジニア、TensorFlow専門家 | Task: 要件1に基づいて既存の64次元Two-Towerアーキテクチャを768次元に拡張し、PostgreSQLのvector(768)テーブルとの完全互換性を実現 | Restrictions: 既存のモデル構造パターン維持、過学習防止、推論速度500ms未満の保持 | Success: 768次元埋め込みが正規化され、コサイン類似度で意味的に適切な関係を表現し、推論性能が要求を満たす_

- [x] 4. 改良された訓練パイプラインの実装
  - File: ml_pipeline/training/production_trainer.py
  - 正例・負例のバランシング最適化（1:2比率）
  - 時系列考慮の訓練・検証データ分割
  - Purpose: 本番環境でのモデル品質と安定性の向上
  - _Leverage: 既存のcreate_training_pairs、early stopping実装_
  - _Requirements: 要件3, 要件4_
  - _Prompt: Role: ML Ops エンジニア、モデル訓練専門家 | Task: 要件3と4に基づいて正例：負例=1:2のバランシング、時系列データ分割、Early Stopping最適化により安定した高精度訓練パイプラインを構築 | Restrictions: 現在の評価指標（AUC-ROC>0.95）以上の性能維持、訓練時間1時間未満、メモリ使用量8GB未満 | Success: 安定してAUC-PR>0.99を達成し、過学習なく汎化性能の高いモデルが生成される_

### 実際のモデル訓練・検証

- [x] 5. Two-Tower モデルの実際の訓練実行
  - File: scripts/train_production_two_tower.py
  - 実装済みパイプラインによる実際のモデル訓練
  - 訓練済みモデルファイル(.keras)の生成
  - Purpose: 実装コードから実際に使用可能なモデルの生成
  - _Leverage: 既存のproduction_trainer.py、enhanced_two_tower_trainer.py_
  - _Requirements: 要件3, 要件4_
  - _Prompt: Role: ML実行エンジニア、モデル訓練専門家 | Task: 実装済みの訓練パイプラインを使用して実際にTwo-Towerモデルを訓練し、768次元の訓練済みモデルファイルを生成 | Restrictions: 訓練時間1時間未満、メモリ使用量8GB未満、既存データ構造との互換性維持 | Success: user_tower.keras と item_tower.keras が正常に生成され、AUC-PR>0.99の性能を達成し、推論可能な状態で保存される_

### 基本テスト・バリデーション

- [x] 6. パフォーマンステスト・ベンチマークスイートの作成
  - File: tests/performance/two_tower_benchmark_suite.py
  - 訓練・推論・埋め込み更新の性能基準テスト
  - メモリ・計算時間の制約確認
  - Purpose: 性能要件の継続的な検証
  - _Leverage: 既存のテスト構造とベンチマークパターン_
  - _Requirements: 要件4_
  - _Prompt: Role: パフォーマンステストエンジニア、ベンチマーク専門家 | Task: 要件4の性能基準（訓練1時間未満、推論500ms未満、メモリ制約）を継続的に検証する包括的なベンチマークスイートを作成 | Restrictions: テスト実行時間の最適化、本番環境への影響回避、再現可能な性能測定 | Success: 全ての性能要件が自動テストで検証され、性能劣化が早期発見され、CI/CDパイプラインに統合される_

## フェーズ2: システム統合

### 埋め込み管理システム

- [x] 7. バッチ埋め込み更新システムの強化
  - File: ml_pipeline/preprocessing/production_embedding_updater.py
  - 20万ビデオ・1万ユーザー規模への対応
  - PostgreSQLとの効率的なUPSERT処理
  - Purpose: 大規模データでの埋め込み更新の効率化
  - _Leverage: 既存のbatch_embedding_update.py、generate_embeddingsパターン_
  - _Requirements: 要件5_
  - _Prompt: Role: データベースエンジニア、大規模データ処理専門家 | Task: 要件5に基づいて20万ビデオ・1万ユーザーの768次元埋め込みを効率的にバッチ更新し、PostgreSQLとの安全な同期を実現 | Restrictions: 更新処理10分未満、トランザクション整合性保証、既存サービスへの影響最小化 | Success: 大規模埋め込み更新が安定動作し、データベース同期が正確に行われ、推薦サービスが中断されない_

- [x] 8. リアルタイム埋め込み生成APIの実装
  - File: ml_pipeline/inference/realtime_embedding_generator.py
  - 新規ユーザー・新規ビデオの即座埋め込み生成
  - コールドスタート問題への対応
  - Purpose: 動的な埋め込み生成による推薦サービスの継続性確保
  - _Leverage: 既存のuser_tower、item_tower推論パターン_
  - _Requirements: 要件2, 要件5_
  - _Prompt: Role: リアルタイムシステムエンジニア、API開発専門家 | Task: 要件2と5に基づいて新規ユーザー・ビデオに対するリアルタイム埋め込み生成APIを実装し、コールドスタート問題を解決 | Restrictions: レスポンス時間500ms未満、デフォルト埋め込みの品質保証、既存システムとの互換性維持 | Success: 新規エンティティに対して意味のある埋め込みが即座に生成され、推薦品質の大幅な低下なく推薦サービスが継続される_

### システム統合テスト

- [x] 9. 統合テスト・品質検証システムの実装
  - File: tests/integration/two_tower_full_pipeline_test.py
  - E2E パイプライン（データ→訓練→埋め込み→推論→評価）の統合テスト
  - 品質回帰テストの自動化
  - Purpose: システム全体の信頼性と品質の継続保証
  - _Leverage: 既存のテストパターンとCI/CD構造_
  - _Requirements: 全要件_
  - _Prompt: Role: QAエンジニア、統合テスト専門家 | Task: 全要件を満たすTwo-Towerシステムの完全なE2Eテストスイートを構築し、品質回帰を自動検出できるようにする | Restrictions: テスト実行時間の最適化、テスト環境の独立性確保、本番データとの明確な分離 | Success: 全システムコンポーネントが統合テストで検証され、品質劣化が自動検出され、継続的インテグレーションパイプラインが確立される_

## フェーズ3: 本番環境準備

### 本番環境統合

- [x] 10. Edge Functions推論APIの最適化
  - File: supabase/functions/enhanced_two_tower_recommendations/index.ts
  - 768次元ベクター対応の推論最適化
  - 推薦理由の生成と多様性確保
  - Purpose: 本番環境での高速・高品質推薦配信
  - _Leverage: 既存のtwo_tower_recommendations Edge Function_
  - _Requirements: 要件5_
  - _Prompt: Role: サーバーレスエンジニア、Edge Functions専門家 | Task: 要件5に基づいて既存Edge Functionを768次元ベクターに対応させ、推薦理由付きで500ms未満の高速推薦配信を実現 | Restrictions: Edge Functions制約内での動作、メモリ使用量最小化、既存APIとの互換性維持 | Success: 768次元ベクター推論が高速動作し、推薦理由付きで多様性の保たれた推薦結果が返却される_

### 監視・評価システム

- [x] 11. 推薦品質監視システムの実装
  - File: ml_pipeline/monitoring/recommendation_quality_monitor.py
  - リアルタイムでの推薦精度・多様性・エンゲージメント追跡
  - アラート・ダッシュボード機能
  - Purpose: 推薦システムの健全性の継続監視
  - _Leverage: 既存の評価指標計算パターン_
  - _Requirements: 要件6_
  - _Prompt: Role: MLOps監視エンジニア、データ分析専門家 | Task: 要件6に基づいて推薦システムの精度、多様性、ユーザーエンゲージメントをリアルタイムで監視し、品質劣化を早期検出するシステムを構築 | Restrictions: 監視処理のオーバーヘッド最小化、プライバシー保護、異常検出の精度確保 | Success: 推薦品質の変化が迅速に検出され、問題発生時に適切なアラートが生成され、データドリブンな改善判断が可能_

### データ統合・移行

- [x] 12. 疑似ユーザーからリアルユーザーへの移行戦略実装
  - File: ml_pipeline/migration/pseudo_to_real_user_migration.py
  - 段階的なデータ移行とモデル再訓練
  - 品質劣化の監視と対策
  - Purpose: 評価ベースから実際のユーザー行動ベースへの安全な移行
  - _Leverage: 既存の疑似ユーザーデータ処理パターン_
  - _Requirements: 要件2, 要件6_
  - _Prompt: Role: データマイグレーションエンジニア、MLパイプライン専門家 | Task: 要件2と6に基づいて現在の評価ベース疑似ユーザーデータ（99.7% AUC-PR）からリアルユーザーデータへの段階的移行戦略を実装 | Restrictions: 推薦品質の大幅低下回避、サービス継続性確保、移行過程の可視化 | Success: データ移行が計画通り実行され、推薦品質が段階的に向上し、最終的にリアルユーザーベースで現在以上の性能を達成_

## フェーズ4: 運用・管理 (最後)

### 運用管理システム

- [x] 13. モデル版管理・A/Bテストシステムの実装
  - File: ml_pipeline/deployment/model_versioning_system.py
  - 複数モデルバージョンの並行運用
  - 段階的ロールアウトとロールバック機能
  - Purpose: 安全なモデル更新とパフォーマンス比較
  - _Leverage: 既存のモデルエクスポート・メタデータ管理パターン_
  - _Requirements: 要件6_
  - _Prompt: Role: MLOpsエンジニア、DevOps専門家 | Task: 要件6に基づいて複数Two-Towerモデルバージョンの安全な並行運用、A/Bテスト、段階的デプロイメントシステムを構築 | Restrictions: 本番サービスへの影響最小化、迅速なロールバック対応、パフォーマンス監視の継続 | Success: モデル更新が安全に実行され、A/Bテストにより推薦品質向上が定量的に測定可能_