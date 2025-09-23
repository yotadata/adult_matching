# Requirements Document

## Introduction

バックエンド全体のリファクタリングは、現在の煩雑なフォルダ構造と未使用ファイルの整理、そして保守性・拡張性の向上を目的とします。この取り組みにより、開発効率の向上、新機能追加の容易性、及びシステム全体の理解しやすさを実現します。

現在のバックエンドは段階的な開発により複数のアプローチが混在し、重複機能や未使用コードが存在しています。リファクタリングにより、一貫性のあるアーキテクチャと明確な責任分離を確立します。

## Alignment with Product Vision

このリファクタリングは以下の製品ビジョンを支援します：

- **スケーラブルアーキテクチャ**: 大規模データベース（200k+動画）をサポートする構造的基盤の強化
- **データ駆動**: ML推薦システムの効率的な統合とデータ処理パイプラインの最適化
- **プライバシーファースト**: RLSポリシーとセキュリティの一貫した実装
- **モバイル最適化**: 高速レスポンス（<500ms）を実現するバックエンド効率化

## Requirements

### Requirement 1: Edge Functions統合とクリーンアップ

**User Story:** 開発者として、重複するEdge Functionsを統合し、現在使用されていない関数を削除したい。これにより、保守コストを削減し、デプロイメント効率を向上させる。

#### Acceptance Criteria

1. WHEN 現在のEdge Functions使用状況を調査 THEN システム SHALL 実際に使用されている関数のみを特定する
2. WHEN 重複機能を持つ関数を発見 THEN システム SHALL 最新・最適な実装に統合する
3. WHEN 未使用関数を特定 THEN システム SHALL 安全に削除し、デプロイメントサイズを最適化する
4. IF enhanced_two_tower_recommendations が推薦機能の標準 THEN システム SHALL 古い recommendations と two_tower_recommendations を非推奨化する

### Requirement 2: MLパイプライン構造の最適化

**User Story:** データサイエンティストとして、MLパイプラインの構造を明確化し、トレーニングから推論までの一貫したワークフローを確立したい。これにより、モデル開発とデプロイメントの効率を向上させる。

#### Acceptance Criteria

1. WHEN MLパイプライン構造を整理 THEN システム SHALL training/preprocessing/inference/deployment の明確な分離を実現する
2. WHEN モデルアーティファクトを管理 THEN システム SHALL バージョン管理された格納構造を提供する
3. WHEN 重複するトレーニングスクリプトを統合 THEN システム SHALL 統一されたトレーニングインターフェースを提供する
4. IF 768次元実装が標準 THEN システム SHALL 古い64次元関連ファイルを適切にアーカイブする

### Requirement 3: データ処理パイプライン統合

**User Story:** データエンジニアとして、分散している類似のデータ処理機能を統合し、一貫したデータフロー管理を実現したい。これにより、データ品質の向上と処理効率の最適化を図る。

#### Acceptance Criteria

1. WHEN 重複するスクリプト機能を特定 THEN システム SHALL 機能統合と重複削除を実行する
2. WHEN DMM API統合スクリプトを整理 THEN システム SHALL 推奨アプローチ（Node.js）と非推奨アプローチを明確化する
3. WHEN データ処理ワークフローを標準化 THEN システム SHALL raw→processed→training の一貫したフローを確立する
4. IF batch処理とreal-time処理が混在 THEN システム SHALL 明確な使い分けガイドラインを提供する

### Requirement 4: ディレクトリ構造の合理化

**User Story:** 新規開発者として、プロジェクト構造を理解しやすく、適切なファイル配置場所を容易に特定できるディレクトリ構造を求める。これにより、開発の学習コストを削減し、生産性を向上させる。

#### Acceptance Criteria

1. WHEN ディレクトリ構造を再設計 THEN システム SHALL 機能別の論理的グループ化を実現する
2. WHEN 未使用ファイルを特定 THEN システム SHALL 安全な削除または適切なアーカイブを実行する
3. WHEN ファイル命名規則を統一 THEN システム SHALL 一貫した命名パターンを適用する
4. IF 開発環境とプロダクション環境の混在ファイルが存在 THEN システム SHALL 適切な分離を実現する

### Requirement 5: 包括的テスティング統合

**User Story:** QAエンジニアとして、リファクタリング後のシステムが正常に動作することを検証し、回帰バグの防止を確実にしたい。

#### Acceptance Criteria

1. WHEN リファクタリングを実行 THEN システム SHALL ≥95%のコードカバレッジを維持する
2. WHEN 統合テストを実行 THEN システム SHALL エンドツーエンド機能の正常動作を検証する
3. WHEN パフォーマンステストを実行 THEN システム SHALL <500ms推薦レスポンス要件を満たす
4. IF MLパイプラインをリファクタリング THEN システム SHALL モデル精度（AUC-PR ≥ 0.85）を維持する

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility Principle**: 各ディレクトリ・ファイルは明確に定義された単一の責任を持つ
- **Modular Design**: Edge Functions、MLパイプライン、データ処理の完全な分離と再利用性
- **Dependency Management**: 循環依存の排除と明確な依存関係階層の確立
- **Clear Interfaces**: 各モジュール間の契約と API の明確な定義

### Performance

- **推薦システム**: リファクタリング後も<500ms応答時間を維持
- **MLパイプライン**: トレーニング効率の改善または維持（<2時間/100kサンプル）
- **デプロイメント**: Edge Functions デプロイ時間の20%以上短縮
- **メモリ使用量**: リファクタリング後のピークメモリ使用量<4GB

### Security

- **RLS Policy**: 統合後もRow Level Securityポリシーの完全性維持
- **API セキュリティ**: Edge Functions における入力検証とレート制限の継続
- **認証**: Supabase Auth統合の一貫性とセキュリティ要件遵守
- **データ保護**: 成人向けコンテンツの適切な取り扱いとプライバシー保護

### Reliability

- **後方互換性**: 既存フロントエンド機能への影響なし
- **グレースフル移行**: 段階的リファクタリングによるダウンタイム最小化
- **エラーハンドリング**: 統合後の堅牢なエラー処理とログ記録
- **データ整合性**: リファクタリング過程でのデータ損失ゼロ

### Usability

- **開発者体験**: 明確なドキュメントとセットアップ手順の提供
- **保守性**: 統一されたコーディング規約と構造による保守コスト削減
- **拡張性**: 新機能追加時の明確な追加場所とパターンの確立
- **監視**: リファクタリング後のシステム健全性監視の継続