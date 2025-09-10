# タスク文書

<!-- AI Instructions: For each task, generate a _Prompt field with structured AI guidance following this format:
_Prompt: Role: [specialized developer role] | Task: [clear task description with context references] | Restrictions: [what not to do, constraints] | Success: [specific completion criteria]_
This helps provide better AI agent guidance beyond simple "work on this task" prompts. -->

## 現在実装済みの確認と最適化

- [ ] 1. 既存スキーマの性能最適化確認
  - File: supabase/migrations/20250909000001_optimize_existing_indexes.sql
  - 現在のテーブル（videos, user_embeddings, video_embeddings, user_video_decisions）のインデックス最適化
  - 768次元ベクター検索の性能確認
  - Purpose: 既存実装の性能ボトルネックの解消
  - _Leverage: 既存のテーブル構造とpgvector実装_
  - _Requirements: 要件2, 要件4_
  - _Prompt: Role: データベースパフォーマンスチューナー、PostgreSQL専門家 | Task: 要件2と4に基づいて現在実装済みの768次元ベクターインデックスと頻用クエリのパフォーマンスを分析し、必要な最適化を実装 | Restrictions: 既存データの変更禁止、現在の768次元構造の維持、実行中システムへの影響最小化 | Success: ベクター検索が500ms未満で動作し、推薦クエリの性能が向上する_

## 不足している機能の実装

- [ ] 2. ベクター埋め込み更新関数の作成
  - File: supabase/migrations/20250909000002_embedding_functions.sql
  - user_embeddingsとvideo_embeddingsの効率的な更新関数実装
  - バッチ更新処理の最適化（768次元対応）
  - Purpose: ML埋め込みの効率的な管理
  - _Leverage: 既存のSupabase Functions構造_
  - _Requirements: 要件2_
  - _Prompt: Role: データベース関数開発者、PL/pgSQL専門家 | Task: 要件2に従って既存の768次元user_embeddingsとvideo_embeddingsテーブルの効率的な更新関数を実装し、バッチ処理とエラーハンドリングを含める | Restrictions: 関数実行時間を1秒未満に制限、既存データ構造の維持、メモリ使用量の最適化 | Success: バッチ埋め込み更新が正常動作し、大量データ処理時も安定動作する_

- [ ] 3. 自動タイムスタンプ更新トリガーの設定
  - File: supabase/migrations/20250909000003_timestamp_triggers.sql
  - user_embeddings, video_embeddingsのupdated_at自動更新
  - 全テーブルへの適用確認
  - Purpose: データ変更追跡の自動化
  - _Leverage: PostgreSQLトリガー標準パターン_
  - _Requirements: 要件5_
  - _Prompt: Role: データベース管理者、トリガー専門家 | Task: 要件5に基づいて既存テーブルのupdated_atカラム自動更新トリガーを確認し、不足部分を実装してデータ変更の正確な追跡を実現 | Restrictions: 既存データの変更なし、トリガー処理の高速化、循環参照の回避 | Success: 全テーブルでデータ更新時にタイムスタンプが自動更新され、変更履歴の追跡が正確に行われる_

- [ ] 4. データ整合性チェック関数の作成
  - File: supabase/migrations/20250909000004_data_validation.sql
  - 外部キー整合性検証関数の実装
  - user_video_decisions, embeddingsテーブルの整合性確認
  - Purpose: データ品質の維持
  - _Leverage: PostgreSQL標準検証パターン_
  - _Requirements: 要件1, 要件5_
  - _Prompt: Role: データ品質エンジニア、PostgreSQL専門家 | Task: 要件1と5に従って現在実装されているテーブル間のデータ整合性チェック関数を実装し、外部キー違反や重複データを検出・報告 | Restrictions: 本番データを変更しない、検証処理の軽量化、false positiveの最小化 | Success: データ不整合が正確に検出され、整合性レポートが生成される_

## パフォーマンス向上

- [ ] 5. 推薦システム用複合インデックスの作成
  - File: supabase/migrations/20250909000005_recommendation_indexes.sql
  - user_video_decisionsテーブルの頻用クエリ最適化
  - ベクター検索と組み合わせた複合インデックス
  - Purpose: 推薦クエリのレスポンス時間短縮
  - _Leverage: 現在の推薦システムクエリパターン_
  - _Requirements: 要件2, 要件4_
  - _Prompt: Role: データベースパフォーマンスチューナー、インデックス専門家 | Task: 要件2と4に基づいて現在のuser_video_decisionsテーブルと768次元ベクター検索を組み合わせた推薦クエリ用の最適化インデックスを作成 | Restrictions: インデックスサイズの制限、書き込み性能への影響最小化、既存クエリとの互換性維持 | Success: 推薦クエリが500ms未満で実行され、全体的なDB性能が向上する_

- [ ] 6. バッチ処理最適化の実装
  - File: supabase/migrations/20250909000006_batch_optimization.sql
  - 大量ベクターデータ処理用の最適化設定
  - メモリ使用量の制御（768次元対応）
  - Purpose: MLパイプラインとの効率的連携
  - _Leverage: 現在のMLパイプライン連携パターン_
  - _Requirements: 要件2_
  - _Prompt: Role: データベースアーキテクト、バッチ処理専門家 | Task: 要件2に従って現在の768次元ベクターデータを含む大量データ処理（20万件以上のビデオデータ）用の最適化設定を実装し、MLパイプラインとの効率的な連携を実現 | Restrictions: 通常操作への影響回避、既存768次元構造の維持、処理時間の予測可能性確保 | Success: 大量データ処理が安定動作し、MLパイプラインとの連携が効率的に行われる_

## テストとモニタリング

- [ ] 7. データベーススキーマテストの作成
  - File: tests/database/schema.test.sql
  - 現在実装のテーブル構造とリレーションシップのテスト
  - 制約とトリガーの動作確認
  - Purpose: 既存スキーマ実装の正確性検証
  - _Leverage: pgTAPテストフレームワーク_
  - _Requirements: 全要件_
  - _Prompt: Role: データベーステストエンジニア、pgTAP専門家 | Task: 全要件をカバーする現在実装済みのデータベーススキーマテストを作成し、テーブル構造、制約、RLSポリシーの正確な動作を検証 | Restrictions: テストデータによる本番環境汚染回避、テスト実行時間の最適化、テスト間の独立性確保 | Success: 全てのスキーマ要素が仕様通り動作し、回帰テストが可能な状態になる_

- [ ] 8. パフォーマンステストの実装
  - File: tests/database/performance.test.sql
  - 768次元ベクター検索のベンチマークテスト
  - user_video_decisionsを含む推薦クエリの負荷テスト
  - Purpose: 性能要件の検証
  - _Leverage: 現在のデータ規模とクエリパターン_
  - _Requirements: 要件2, 要件4_
  - _Prompt: Role: パフォーマンステストエンジニア、ベンチマーク専門家 | Task: 要件2と4のパフォーマンス基準に基づいて現在の768次元ベクター検索とuser_video_decisionsテーブルを使用した推薦クエリのベンチマークテストを実装 | Restrictions: 本番環境への影響回避、現在のデータ規模での適切なテスト、テスト結果の再現性確保 | Success: パフォーマンス要件（500ms未満）が満たされ、現在の実装での負荷テストが安定する_

## 運用改善

- [ ] 9. 既存マイグレーション整理スクリプトの作成
  - File: scripts/consolidate-migrations.sh
  - 複数回のマイグレーション履歴の整理と文書化
  - 現在の実装状況のドキュメント生成
  - Purpose: 開発チームの理解促進と保守性向上
  - _Leverage: 現在のマイグレーション履歴_
  - _Requirements: 要件6_
  - _Prompt: Role: DevOpsエンジニア、データベース管理専門家 | Task: 要件6に基づいて現在の複数マイグレーション履歴を整理し、実装状況のドキュメントを生成して開発チームの理解を促進 | Restrictions: 既存マイグレーション履歴の保持、本番データへの影響回避、文書の正確性確保 | Success: 現在の実装状況が明確に文書化され、開発チームが容易に理解できる状態になる_

- [ ] 10. 型定義の更新確認
  - File: frontend/lib/database.types.ts
  - 現在のスキーマ（user_video_decisions等）に対応した型定義確認
  - フロントエンドとの型安全性確保
  - Purpose: TypeScript統合の完全性確認
  - _Leverage: 現在のSupabase型生成機能_
  - _Requirements: 要件1_
  - _Prompt: Role: TypeScript開発者、型システム専門家 | Task: 要件1に基づいて現在実装されているスキーマ（user_video_decisionsテーブル等）に対応したTypeScript型定義を確認・更新し、フロントエンドとの完全な型安全性を確保 | Restrictions: 既存フロントエンドコードとの互換性維持、自動生成型定義の活用、手動型定義の最小化 | Success: 現在のデータベーススキーマが正確にTypeScript型に反映され、フロントエンドとの型安全性が保たれる_