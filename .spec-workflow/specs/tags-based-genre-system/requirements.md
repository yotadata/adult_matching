# Requirements Document

## Introduction

このspecは、現在`videos.genre`カラムを参照している既存コードを、Supabaseのデータベースに即してtagsテーブルベースの情報取得に置き換えることを目的としています。まずは基本的な動作を確保し、現在のシステムを動作させることを最優先とします。

## Alignment with Product Vision

この機能は、現在のシステムを最小限の変更で動作可能にすることを目標とします：

- **System Continuity**: 既存のgenre参照機能の継続
- **Database Compatibility**: Supabaseの実際のスキーマに適合
- **Minimal Changes**: 最小限のコード変更でシステム復旧

## Requirements

### Requirement 1: 既存genre参照コードの置き換え

**User Story:** As a システム開発者, I want 現在の`videos.genre`を参照するコードを、tagsテーブルベースに置き換える, so that 本番環境で動作するシステムが構築できる

#### Acceptance Criteria

1. WHEN 既存コードが`videos.genre`を参照する時 THEN システム SHALL video_tagsとtagsテーブルから同等の情報を取得する
2. IF Edge Functionsでgenre情報が必要な時 THEN システム SHALL タグベースの代替クエリを実行する
3. WHEN RPC関数でジャンル情報が必要な時 THEN システム SHALL既存の戻り値形式を維持しつつタグから情報を取得する

### Requirement 2: 最小限のシステム変更

**User Story:** As a システム開発者, I want 最小限のコード変更でシステムを復旧させる, so that 迅速にサービスを動作可能にできる

#### Acceptance Criteria

1. WHEN 既存の推薦システムロジックを修正する時 THEN システム SHALL 既存の関数シグネチャを維持する
2. IF データベーススキーマの変更が必要な時 THEN システム SHALL 既存のtagsテーブル構造を活用する
3. WHEN フロントエンドのAPI呼び出しがある時 THEN システム SHALL 既存のレスポンス形式を保持する

## Non-Functional Requirements

### Performance

- タグベースクエリ: 既存システムと同等の性能を維持
- 既存のインデックス構造を活用した検索性能確保

### Compatibility

- 既存のEdge Functions（enhanced_two_tower_recommendations等）との完全互換性
- 現在のRPC関数（get_personalized_videos_feed等）のAPI仕様維持
- フロントエンドからのAPI呼び出しに対する既存レスポンス形式の保持

### Reliability

- タグデータが存在しない場合のフォールバック動作定義
- エラー発生時の既存システム動作継続