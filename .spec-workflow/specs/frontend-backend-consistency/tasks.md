# Tasks Document

## 前提確認
既存のコード解析ツール（`scripts/utilities/ast_grep.js`）が存在するため、これを拡張してSupabase整合性チェック機能を追加する汎用的なアプローチを採用。

- [x] 1. 既存AST解析ツールの拡張によるSupabase解析機能追加
  - File: scripts/utilities/supabase-analyzer.js（既存ast_grep.jsを拡張）
  - 既存のAST解析インフラを活用してSupabase専用の解析ルールを追加
  - Edge Functions、RPC関数、データベーススキーマの解析機能を統合
  - Purpose: 既存ツールを拡張し、汎用性の高いSupabase解析機能を提供
  - _Leverage: scripts/utilities/ast_grep.js, scripts/utilities/utils.js_
  - _Requirements: 1.1, 1.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: DevOps Engineer specializing in code analysis tool development and AST parsing | Task: Extend existing AST analysis infrastructure (scripts/utilities/ast_grep.js) to add Supabase-specific analysis capabilities following requirements 1.1 and 1.2, creating a reusable and generic framework | Restrictions: Do not break existing ast_grep functionality, maintain backward compatibility, ensure the tool can be reused for other projects | _Leverage: scripts/utilities/ast_grep.js infrastructure and patterns | _Requirements: 1.1 (environment analysis), 1.2 (schema consistency) | Success: Extended tool maintains existing functionality while adding Supabase analysis, framework is reusable for other Supabase projects, outputs structured analysis results | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 2. Edge Functions レスポンス形式バリデーターの実装
  - File: scripts/utilities/edge-function-validator.js（既存utils.jsパターンを活用）
  - 全Edge Functionsのレスポンス形式とエラーハンドリングを検証
  - videos-feed, likes, update_user_embedding, delete_accountのレスポンス統一確認
  - Purpose: Edge Functionsの一貫したレスポンス形式を保証（汎用的なSupabase Edge Function検証ツール）
  - _Leverage: 既存のSupabase認証設定、scripts/utilities/utils.js_
  - _Requirements: 2.1, 2.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Backend Developer with expertise in API validation and Supabase Edge Functions | Task: Create edge-function-validator.js leveraging existing scripts/utilities/utils.js patterns to verify response formats and error handling across all Edge Functions following requirements 2.1 and 2.2, designed as reusable tool for any Supabase project | Restrictions: Do not modify existing Edge Functions code, perform validation through API calls only, maintain environment compatibility, ensure tool reusability | _Leverage: existing Supabase auth configuration, scripts/utilities/utils.js patterns | _Requirements: 2.1 (response format standardization), 2.2 (error handling consistency) | Success: Validator detects response format inconsistencies, verifies error handling patterns, generates actionable recommendations, tool is reusable for other Supabase projects | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 3. データベーススキーマ整合性チェッカーの作成
  - File: scripts/utilities/database-schema-checker.js（既存パターンを汎用化）
  - RPC関数定義とEdge Functions使用の整合性を確認
  - get_videos_feed, get_user_likes, get_user_liked_tags, get_user_liked_performers関数の検証
  - Purpose: データベースRPC関数とEdge Functions間の整合性確保（汎用PostgreSQL+Supabase検証ツール）
  - _Leverage: 既存のマイグレーションファイルパターン、scripts/utilities/*のコード解析手法_
  - _Requirements: 3.1, 3.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Database Engineer with expertise in PostgreSQL and RPC function analysis | Task: Create database-schema-checker.js to verify consistency between RPC function definitions and Edge Functions usage following requirements 3.1 and 3.2, designed as reusable tool for any PostgreSQL+Supabase project | Restrictions: Do not modify database schema, perform read-only analysis, ensure compatibility with existing migrations, make tool project-agnostic | _Leverage: existing migration file patterns, scripts/utilities/* code analysis methods | _Requirements: 3.1 (RPC function consistency), 3.2 (Edge Functions usage verification) | Success: Checker identifies RPC function inconsistencies, validates parameter matching, detects unused or missing functions, tool works with any Supabase project | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 4. API レスポンス実環境バリデーターの実装
  - File: scripts/utilities/api-response-validator.js（既存Supabaseクライアント設定を活用）
  - 実際のAPIエンドポイントを呼び出してレスポンス形式を検証
  - 開発環境と本番環境での整合性確認
  - Purpose: 実環境でのAPI動作とレスポンス形式の検証（汎用API検証ツール）
  - _Leverage: 既存のSupabase環境設定、frontend/src/lib/supabase.ts設定パターン_
  - _Requirements: 4.1, 4.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in API testing and environment validation | Task: Create api-response-validator.js to test actual API endpoints and verify response formats following requirements 4.1 and 4.2, designed as reusable tool for any API testing project | Restrictions: Use test data only, do not affect production data, ensure environment isolation, make tool configurable for different APIs | _Leverage: existing Supabase environment configuration, frontend/src/lib/supabase.ts patterns | _Requirements: 4.1 (live API testing), 4.2 (response format verification) | Success: Validator tests all API endpoints, verifies response schemas, detects environment inconsistencies, tool is reusable for other API projects | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 5. 認証整合性チェッカーの作成
  - File: scripts/utilities/authentication-checker.js（既存認証パターンを汎用化）
  - Edge FunctionsとRLSポリシー間の認証整合性を確認
  - JWT認証ヘッダー処理の統一性検証
  - Purpose: 認証とセキュリティの一貫性を保証（汎用認証整合性ツール）
  - _Leverage: 既存のRLSポリシー定義、Supabase認証設定パターン_
  - _Requirements: 5.1, 5.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Security Engineer with expertise in authentication systems and RLS policies | Task: Create authentication-checker.js to verify consistency between Edge Functions and RLS policies following requirements 5.1 and 5.2, designed as reusable tool for any Supabase authentication project | Restrictions: Do not modify security configurations, perform read-only security analysis, maintain access control integrity, ensure tool portability | _Leverage: existing RLS policy definitions, Supabase auth configuration patterns | _Requirements: 5.1 (auth header consistency), 5.2 (RLS policy alignment) | Success: Checker identifies auth inconsistencies, validates JWT handling, verifies RLS policy coverage, tool works with any Supabase auth setup | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 6. 統合レポートジェネレーターの実装
  - File: scripts/utilities/report-generator.js（既存レポートパターンを汎用化）
  - 全検証結果を統合した包括的なレポートを生成
  - JSON、HTML、Markdown形式でのレポート出力
  - Purpose: 検証結果の可視化と修正提案の提供（汎用レポートシステム）
  - _Leverage: 既存のSupabaseプロジェクト構造、scripts/ディレクトリのレポート出力パターン_
  - _Requirements: 6.1, 6.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: DevOps Engineer with expertise in reporting systems and data visualization | Task: Create report-generator.js to produce comprehensive validation reports following requirements 6.1 and 6.2, designed as reusable tool for any validation project | Restrictions: Ensure report security (no sensitive data exposure), maintain report format consistency, optimize for readability, make tool configurable | _Leverage: existing Supabase project structure, scripts/ directory report output patterns | _Requirements: 6.1 (comprehensive reporting), 6.2 (actionable recommendations) | Success: Generator produces clear, actionable reports in multiple formats, identifies priorities for fixes, provides implementation guidance, tool is reusable for other projects | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 7. CLI インターフェースの作成
  - File: scripts/supabase-consistency-checker.js（既存scriptsパターンを活用）
  - コマンドライン実行インターフェースの実装
  - 設定ファイル対応と実行オプションの提供
  - Purpose: 開発者が簡単にツールを実行できる環境を提供（汎用CLIツール）
  - _Leverage: 既存のプロジェクト構造パターン、scripts/ディレクトリのCLIパターン_
  - _Requirements: 7.1_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Developer Tools Engineer with expertise in CLI design and user experience | Task: Create supabase-consistency-checker.js for command-line interface following requirement 7.1, designed as reusable CLI tool for any Supabase project | Restrictions: Maintain CLI usability standards, provide clear help documentation, ensure cross-platform compatibility, make tool configurable and portable | _Leverage: existing project structure patterns, scripts/ directory CLI patterns | _Requirements: 7.1 (CLI interface design) | Success: CLI is intuitive and well-documented, supports configuration files, provides clear output and error messages, tool works with any Supabase project | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 8. 設定ファイルシステムの実装
  - File: scripts/config/supabase-consistency.config.js（既存設定パターンを汎用化）
  - プロジェクト設定、環境設定、検証ルールの管理
  - 環境別設定（開発、本番）の対応
  - Purpose: ツールの設定可能性と環境適応性を提供（汎用設定システム）
  - _Leverage: 既存の環境設定パターン、supabase/config.toml設定構造_
  - _Requirements: 8.1_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Configuration Engineer with expertise in environment management and configuration systems | Task: Create supabase-consistency.config.js for configuration management following requirement 8.1, designed as reusable configuration system for any validation project | Restrictions: Ensure configuration security (no hardcoded secrets), support environment variables, maintain backward compatibility, make configuration portable | _Leverage: existing environment configuration patterns, supabase/config.toml structure | _Requirements: 8.1 (configuration management) | Success: Configuration system is flexible and secure, supports multiple environments, provides clear configuration documentation, works with any project setup | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 9. ユニットテストスイートの作成
  - File: tests/consistency-checker/（既存testsパターンを活用）
  - 全コンポーネントのユニットテスト実装
  - モックデータとテストユーティリティの作成
  - Purpose: ツールの信頼性と品質保証（汎用テストフレームワーク）
  - _Leverage: 既存のテストパターンとモックデータ、tests/ディレクトリ構造_
  - _Requirements: 9.1, 9.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer with expertise in unit testing and test automation frameworks | Task: Create comprehensive unit test suite following requirements 9.1 and 9.2, designed as reusable testing framework for any validation tool | Restrictions: Ensure test isolation, mock external dependencies, maintain test performance, make tests portable | _Leverage: existing test patterns and mock data, tests/ directory structure | _Requirements: 9.1 (comprehensive test coverage), 9.2 (test quality assurance) | Success: Test suite achieves >95% code coverage, all tests run reliably, edge cases are covered, framework is reusable for other projects | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_

- [x] 10. 統合テストとドキュメント作成
  - File: docs/supabase-consistency-checker.md, tests/integration/（既存パターンを活用）
  - 統合テストの実装とツール使用ドキュメントの作成
  - CI/CD パイプライン統合の設定
  - Purpose: ツールの完全な動作検証と使用ガイドの提供（汎用ドキュメント・テストフレームワーク）
  - _Leverage: 既存のCI/CDパターンとドキュメント構造、docs/仕様書パターン_
  - _Requirements: 10.1, 10.2_
  - _Prompt: Implement the task for spec frontend-backend-consistency, first run spec-workflow-guide to get the workflow guide then implement the task: Role: DevOps Engineer with expertise in integration testing and technical documentation | Task: Create integration tests and comprehensive documentation following requirements 10.1 and 10.2, designed as reusable documentation and testing framework for any validation tool | Restrictions: Ensure integration test reliability, maintain documentation accuracy, support CI/CD automation, make framework portable | _Leverage: existing CI/CD patterns and documentation structure, docs/ specification patterns | _Requirements: 10.1 (integration testing), 10.2 (comprehensive documentation) | Success: Integration tests validate end-to-end functionality, documentation is clear and complete, CI/CD integration works seamlessly, framework is reusable for other projects | Instructions: Mark this task as in-progress [-] in tasks.md before starting, then mark as complete [x] when finished_