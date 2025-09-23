# Tasks Document

## Phase 1: クエリ構造の根本変更

- [x] 1. 標準クエリパターンの確立
  - File: supabase/functions/_shared/query-patterns.ts
  - 全videosクエリにtags JOINを標準化するクエリパターンを定義
  - genreカラム参照を完全に除去したクエリテンプレートを作成
  - Purpose: 統一されたtags JOINパターンの確立
  - _Leverage: 既存のSupabaseクライアントパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Database Developer specializing in PostgreSQL and Supabase query optimization | Task: Create standardized query patterns for videos with tags JOIN replacing genre column references, defining reusable query templates for all Edge Functions following requirements 1.1 and 1.2 | Restrictions: Do not use genre column, ensure Unknown fallback for missing tags, maintain existing performance | Leverage: Existing Supabase client patterns and join syntax | Success: Query patterns are standardized, all Edge Functions can use common patterns, performance is maintained or improved. After completion, update tasks.md to mark this task as [x] completed._

- [x] 2. TypeScript型定義の根本変更
  - File: supabase/functions/_shared/types.ts
  - Video型からgenreプロパティを削除し、VideoWithTags型を新規作成
  - 全インターフェースをtagsベースに変更
  - Purpose: 型安全性を保ちながらtagsシステムに移行
  - _Leverage: 既存の型定義パターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: TypeScript Developer with expertise in type systems and interface design | Task: Replace Video interface with VideoWithTags interface removing genre property and adding tags-based properties following requirements 1.1 and 1.2 | Restrictions: Maintain backward compatibility where possible, ensure type safety, do not break existing API contracts | Leverage: Existing type definition patterns and interface inheritance | Success: All types compile without errors, VideoWithTags interface is comprehensive, existing code transitions smoothly. After completion, update tasks.md to mark this task as [x] completed._

## Phase 2: 推薦システム根本再構築 (Group A)

- [x] 3. enhanced_two_tower_recommendations/index.ts の完全書き換え
  - File: supabase/functions/enhanced_two_tower_recommendations/index.ts
  - videos.genreの全参照をtags JOINに置換
  - ユーザージャンル嗜好分析をtagsベースに変更
  - Purpose: 主要推薦システムの根本修正
  - _Leverage: 新しいquery-patterns.ts、VideoWithTags型_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer specializing in recommendation systems and PostgreSQL | Task: Completely rewrite enhanced_two_tower_recommendations to use tags-based queries instead of videos.genre, updating user genre preference analysis to use tags system following requirements 1.1 and 1.2 | Restrictions: Maintain existing API response format, ensure Unknown genre fallback, do not break recommendation quality | Leverage: New query-patterns.ts and VideoWithTags types from previous tasks | Success: Recommendation system works with tags, user preferences are correctly analyzed, API responses unchanged. After completion, update tasks.md to mark this task as [x] completed._

- [x] 4. content/recommendations/index.ts の完全書き換え
  - File: supabase/functions/content/recommendations/index.ts
  - 全てのgenre参照をtags JOINクエリに置換
  - ジャンル類似度計算をtagsベースに変更
  - Purpose: コンテンツ推薦システムの根本修正
  - _Leverage: 新しいquery-patterns.ts、VideoWithTags型_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer with expertise in content recommendation algorithms | Task: Rewrite content recommendations to use tags-based queries replacing all videos.genre references and updating genre similarity calculations following requirements 1.1 and 1.2 | Restrictions: Preserve recommendation accuracy, maintain API compatibility, ensure proper error handling | Leverage: Standardized query patterns and VideoWithTags interface | Success: Content recommendations work with tags system, similarity calculations are accurate, performance is maintained. After completion, update tasks.md to mark this task as [x] completed._

- [x] 5. two_tower_recommendations/index.ts の完全書き換え
  - File: supabase/functions/two_tower_recommendations/index.ts
  - genre_encodedをtags_encodedに変更
  - エンベディング生成ロジックをtagsベースに変更
  - Purpose: Two-Towerモデルの根本修正
  - _Leverage: 新しいquery-patterns.ts、VideoWithTags型_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer specializing in embedding systems and neural networks | Task: Rewrite two_tower_recommendations replacing genre_encoded with tags_encoded and updating embedding generation logic to use tags system following requirements 1.1 and 1.2 | Restrictions: Maintain embedding quality, preserve model architecture compatibility, ensure proper encoding | Leverage: Query patterns and type definitions from previous tasks | Success: Two-tower model works with tags, embeddings are properly generated, model performance is maintained. After completion, update tasks.md to mark this task as [x] completed._

## Phase 3: フィード・検索システム (Group B)

- [x] 6. content/feed/index.ts の修正
  - File: supabase/functions/content/feed/index.ts
  - 全てのgenre参照をtags JOINに置換
  - ジャンル多様性計算をtagsベースに変更
  - Purpose: フィードシステムの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer with expertise in content feed algorithms | Task: Update content feed system to use tags-based queries replacing all genre references and updating genre diversity calculations following requirements 1.1 and 1.2 | Restrictions: Maintain feed quality and diversity, preserve API response format, ensure performance | Leverage: Standardized query patterns from task 1 | Success: Feed system works with tags, diversity calculations are accurate, user experience unchanged. After completion, update tasks.md to mark this task as [x] completed._

- [x] 7. content/videos-feed/index.ts の修正
  - File: supabase/functions/content/videos-feed/index.ts
  - genre_filterをtags-based filteringに変更
  - ページネーション機能でのジャンル処理を修正
  - Purpose: ビデオフィードの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer with expertise in pagination and filtering systems | Task: Update videos feed to replace genre_filter with tags-based filtering and fix genre processing in pagination following requirements 1.1 and 1.2 | Restrictions: Maintain pagination performance, preserve filtering functionality, ensure backward compatibility | Leverage: Query patterns and filtering logic from previous tasks | Success: Video feed filtering works with tags, pagination performance maintained, API compatibility preserved. After completion, update tasks.md to mark this task as [x] completed._

- [x] 8. feed_explore/index.ts の修正
  - File: supabase/functions/feed_explore/index.ts
  - 探索フィード生成でのgenre参照をtags JOINに置換
  - ジャンル多様性ロジックをtagsベースに変更
  - Purpose: 探索フィードの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer with expertise in content discovery algorithms | Task: Update explore feed generation to use tags-based queries replacing genre references and updating diversity logic following requirements 1.1 and 1.2 | Restrictions: Maintain exploration quality, preserve discovery effectiveness, ensure proper content variety | Leverage: Standardized patterns and diversity calculations | Success: Explore feed works with tags system, content discovery remains effective, diversity is properly maintained. After completion, update tasks.md to mark this task as [x] completed._

- [x] 9. content/search/index.ts の修正
  - File: supabase/functions/content/search/index.ts
  - 検索クエリでのgenre検索をtags-basedに変更
  - ソート機能でのジャンル処理を修正
  - Purpose: 検索機能の根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer specializing in search algorithms and full-text search | Task: Update search functionality to replace genre search with tags-based search and fix genre sorting following requirements 1.1 and 1.2 | Restrictions: Maintain search accuracy, preserve sorting functionality, ensure search performance | Leverage: Query patterns and search optimization techniques | Success: Search works with tags system, sorting is accurate, search performance maintained. After completion, update tasks.md to mark this task as [x] completed._

- [x] 10. likes/index.ts の修正
  - File: supabase/functions/likes/index.ts
  - ユーザーのいいね履歴でのgenre分析をtagsベースに変更
  - レスポンス形式を維持しつつ内部処理を修正
  - Purpose: いいね機能の根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer with expertise in user interaction analytics | Task: Update likes functionality to use tags-based genre analysis replacing video.genre references while maintaining response format following requirements 1.1 and 1.2 | Restrictions: Preserve API response format, maintain user preference tracking, ensure data consistency | Leverage: Standardized query patterns and analytics logic | Success: Likes system works with tags, user preferences tracked accurately, API responses unchanged. After completion, update tasks.md to mark this task as [x] completed._

- [x] 11. user-management/likes/index.ts の修正
  - File: supabase/functions/user-management/likes/index.ts
  - ユーザー管理でのいいねジャンル分析をtagsベースに変更
  - ユーザープロファイル生成ロジックを修正
  - Purpose: ユーザー管理システムの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer specializing in user management and profile analytics | Task: Update user management likes to use tags-based genre analysis and fix user profile generation logic following requirements 1.1 and 1.2 | Restrictions: Maintain user profile accuracy, preserve management functionality, ensure data privacy | Leverage: Query patterns and user analytics from previous tasks | Success: User management works with tags system, profile generation accurate, functionality preserved. After completion, update tasks.md to mark this task as [x] completed._

## Phase 4: ユーティリティ・管理系 (Group C)

- [x] 12. _shared/content.ts の修正
  - File: supabase/functions/_shared/content.ts
  - 共有コンテンツユーティリティのgenre参照をtags JOINに置換
  - フィルタリング関数をtagsベースに変更
  - Purpose: 共有ユーティリティの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer specializing in shared utilities and helper functions | Task: Update shared content utilities to replace genre references with tags-based queries and update filtering functions following requirements 1.1 and 1.2 | Restrictions: Maintain utility function compatibility, preserve existing behavior, ensure performance | Leverage: Standardized query patterns and utility patterns | Success: Shared utilities work with tags system, functions remain compatible, performance maintained. After completion, update tasks.md to mark this task as [x] completed._

- [x] 13. _shared/feature_preprocessor.ts の修正
  - File: supabase/functions/_shared/feature_preprocessor.ts
  - 特徴量前処理でのgenre_encodedをtags_encodedに変更
  - ジャンルエンコーダーをタグエンコーダーに置換
  - Purpose: ML前処理の根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer specializing in feature engineering and data preprocessing | Task: Update feature preprocessor to replace genre_encoded with tags_encoded and update genre encoder to tags encoder following requirements 1.1 and 1.2 | Restrictions: Maintain feature quality, preserve ML pipeline compatibility, ensure proper encoding | Leverage: Query patterns and encoding techniques | Success: Feature preprocessing works with tags, ML pipeline compatible, encodings are accurate. After completion, update tasks.md to mark this task as [x] completed._

- [x] 14. update_embeddings/index.ts の修正
  - File: supabase/functions/update_embeddings/index.ts
  - エンベディング更新でのgenre特徴量をtagsベースに変更
  - ユーザーエンベディング計算ロジックを修正
  - Purpose: エンベディング更新システムの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer with expertise in embedding systems and vector operations | Task: Update embedding system to replace genre features with tags-based features and fix user embedding calculation logic following requirements 1.1 and 1.2 | Restrictions: Maintain embedding quality, preserve vector dimensions, ensure proper normalization | Leverage: Query patterns and embedding techniques from previous tasks | Success: Embeddings updated with tags features, vector quality maintained, calculations are accurate. After completion, update tasks.md to mark this task as [x] completed._

- [x] 15. update_user_embedding/index.ts の修正
  - File: supabase/functions/update_user_embedding/index.ts
  - ユーザーエンベディング更新でのジャンル分析をtagsベースに変更
  - 嗜好学習アルゴリズムをtagsベースに修正
  - Purpose: ユーザーエンベディング更新の根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer specializing in user modeling and preference learning | Task: Update user embedding system to use tags-based genre analysis and fix preference learning algorithms following requirements 1.1 and 1.2 | Restrictions: Maintain user model accuracy, preserve learning effectiveness, ensure proper preference tracking | Leverage: Standardized patterns and ML techniques | Success: User embeddings work with tags, preference learning accurate, model quality maintained. After completion, update tasks.md to mark this task as [x] completed._

- [x] 16. user-management/embeddings/index.ts の修正
  - File: supabase/functions/user-management/embeddings/index.ts
  - ユーザー管理エンベディングでのジャンル処理をtagsベースに変更
  - 管理インターフェースでの表示ロジックを修正
  - Purpose: ユーザー管理エンベディングの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer specializing in user management and administrative interfaces | Task: Update user management embeddings to use tags-based genre processing and fix display logic following requirements 1.1 and 1.2 | Restrictions: Maintain management interface functionality, preserve administrative capabilities, ensure data accuracy | Leverage: Query patterns and management interface patterns | Success: Management embeddings work with tags, interface functions correctly, data displayed accurately. After completion, update tasks.md to mark this task as [x] completed._

- [x] 17. user-management/_shared/database.ts の修正
  - File: supabase/functions/user-management/_shared/database.ts
  - 共有データベースユーティリティのgenre分析をtagsベースに変更
  - ユーザー嗜好分析関数を修正
  - Purpose: 共有データベースユーティリティの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Database Developer with expertise in shared utilities and user analytics | Task: Update shared database utilities to replace genre analysis with tags-based analysis and fix user preference functions following requirements 1.1 and 1.2 | Restrictions: Maintain utility function reliability, preserve analytics accuracy, ensure database performance | Leverage: Standardized query patterns and database optimization | Success: Database utilities work with tags, analytics functions accurate, performance maintained. After completion, update tasks.md to mark this task as [x] completed._

- [x] 18. user-management/profile/index.ts の修正
  - File: supabase/functions/user-management/profile/index.ts
  - ユーザープロファイルでのfavorite_genresをtagsベースに変更
  - プロファイル生成ロジックを修正
  - Purpose: ユーザープロファイル管理の根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer specializing in user profile management and personalization | Task: Update user profile management to replace favorite_genres with tags-based preferences and fix profile generation logic following requirements 1.1 and 1.2 | Restrictions: Maintain profile accuracy, preserve user preference tracking, ensure profile completeness | Leverage: Query patterns and profile management techniques | Success: User profiles work with tags, preferences tracked accurately, profile generation reliable. After completion, update tasks.md to mark this task as [x] completed._

- [x] 19. recommendations/enhanced_two_tower/index.ts の修正
  - File: supabase/functions/recommendations/enhanced_two_tower/index.ts
  - 拡張Two-Tower推薦でのgenre処理をtagsベースに変更
  - 推薦精度評価ロジックを修正
  - Purpose: 拡張推薦システムの根本修正
  - _Leverage: 標準クエリパターン_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer specializing in advanced recommendation systems and model evaluation | Task: Update enhanced two-tower recommendations to use tags-based genre processing and fix recommendation accuracy evaluation following requirements 1.1 and 1.2 | Restrictions: Maintain recommendation quality, preserve model performance, ensure proper evaluation metrics | Leverage: Standardized patterns and ML evaluation techniques | Success: Enhanced recommendations work with tags, accuracy maintained, evaluation metrics reliable. After completion, update tasks.md to mark this task as [x] completed._

## Phase 5: エンベディングシステム再学習

- [x] 20. 既存エンベディングの無効化と再生成
  - File: scripts/invalidate_embeddings.ts
  - 既存のgenreベースエンベディングを無効化
  - 新しいtagsベースエンベディング生成スクリプトを作成
  - Purpose: エンベディングシステムの完全再構築
  - _Leverage: 既存のエンベディング生成ロジック_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: ML Engineer specializing in embedding systems and data migration | Task: Create script to invalidate existing genre-based embeddings and generate new tags-based embeddings following requirements 1.1 and 1.2 | Restrictions: Ensure data integrity during migration, maintain embedding dimensions, preserve user data | Leverage: Existing embedding generation patterns and migration utilities | Success: Old embeddings safely invalidated, new embeddings generated successfully, system performance maintained. After completion, update tasks.md to mark this task as [x] completed._

## Phase 6: 統合テストと検証

- [x] 21. 統合テストスイートの作成
  - File: tests/integration/tags-genre-system.test.ts
  - 全Edge Functionsの動作を検証するテストを作成
  - タグベースジャンル取得の正確性をテスト
  - Purpose: システム全体の品質保証
  - _Leverage: 既存のテストユーティリティ_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in integration testing and Edge Function testing | Task: Create comprehensive integration test suite to verify all Edge Functions work correctly with tags-based genre system following requirements 1.1 and 1.2 | Restrictions: Test real functionality not implementation details, ensure test reliability, cover error scenarios | Leverage: Existing test utilities and Edge Function testing patterns | Success: All Edge Functions pass integration tests, genre accuracy verified, error handling tested. After completion, update tasks.md to mark this task as [x] completed._

- [x] 22. 本番環境デプロイメントと検証
  - File: scripts/production_deployment.ts
  - 本番環境への段階的デプロイメント実行
  - 全機能の動作確認とパフォーマンステスト
  - Purpose: 本番環境での安全な稼働確保
  - _Leverage: 既存のデプロイメントスクリプト_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec tags-based-genre-system, first run spec-workflow-guide to get the workflow guide then implement the task: Role: DevOps Engineer with expertise in production deployment and system monitoring | Task: Execute staged production deployment and verify all functionality works correctly in production environment following requirements 1.1 and 1.2 | Restrictions: Ensure zero downtime deployment, maintain data integrity, monitor system performance | Leverage: Existing deployment scripts and monitoring tools | Success: Production deployment successful, all functions operational, performance metrics satisfied. After completion, update tasks.md to mark this task as [x] completed._