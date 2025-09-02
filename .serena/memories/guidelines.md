# Development Guidelines and Best Practices

## Architecture Principles
- **Component-Driven Development**: 再利用可能なコンポーネント設計
- **Separation of Concerns**: UI、ロジック、データレイヤーの明確な分離
- **Mobile-First**: レスポンシブデザインはモバイルファーストで実装
- **API-First**: Edge Functions中心のバックエンドAPI設計
- **Security by Design**: Row Level Security (RLS) による適切なデータ保護

## Code Quality Standards
- **TypeScript Strict Mode**: 型安全性を最優先
- **ESLint Compliance**: すべてのコードはESLintルールに準拠
- **Error Handling**: すべてのAPIコールとデータアクセスで適切なエラーハンドリング
- **Loading States**: ユーザー体験向上のためのローディング状態管理
- **Accessibility**: 基本的なアクセシビリティ要件を満たす

## Performance Guidelines
- **Code Splitting**: Next.jsの自動コード分割を活用
- **Image Optimization**: Next.jsのImage componentを使用
- **Vector Search Optimization**: エンベディング計算の効率化
- **Lazy Loading**: 必要に応じて遅延ローディングを実装
- **Caching**: 適切なキャッシュ戦略（Edge Functions、ブラウザキャッシュ）

## Security Considerations
- **Authentication**: すべての認証はSupabase Authを通じて実行
- **RLS Policies**: データベースアクセスはRow Level Securityで制御
- **Input Validation**: ユーザー入力の適切な検証とサニタイゼーション
- **CORS Configuration**: Edge FunctionsでのCORSヘッダー設定
- **Environment Variables**: 機密情報は環境変数で管理

## AI/ML Best Practices
- **Two-Tower Architecture**: ユーザーとアイテムの独立したエンベディング
- **Embedding Normalization**: すべてのベクトルはL2正規化を実行
- **Diversity Consideration**: 推奨結果に多様性を組み込む
- **Cold Start Problem**: 新規ユーザー向けのフォールバック戦略
- **Real-time Updates**: ユーザー行動に基づくリアルタイムエンベディング更新

## User Experience Guidelines
- **Responsive Design**: デスクトップとモバイルで最適化されたレイアウト
- **Animation Standards**: Framer Motionを使用した60fps滑らかアニメーション
- **Gesture Support**: タッチデバイスでの直感的なジェスチャー操作
- **Loading Feedback**: ユーザーに適切なフィードバックを提供
- **Error Messages**: 分かりやすい日本語エラーメッセージ

## Data Management
- **Vector Embeddings**: 768次元ベクトルでの統一された埋め込み表現
- **Batch Processing**: 大量データ処理は適切なバッチサイズで実行
- **Data Migration**: スキーマ変更は適切なマイグレーションスクリプトで管理
- **Backup Strategy**: 重要なユーザーデータのバックアップ戦略
- **GDPR Compliance**: ユーザーデータ削除機能の実装

## Deployment and Operations
- **Environment Separation**: 開発、ステージング、本番環境の明確な分離
- **Version Management**: Edge Functions、ML モデルのバージョン管理
- **Monitoring**: エラー追跡とパフォーマンス監視
- **Rollback Strategy**: 問題発生時の適切なロールバック手順
- **A/B Testing**: 推奨アルゴリズムの継続的な改善

## Documentation Standards
- **Code Comments**: 複雑なロジックには日本語コメント
- **API Documentation**: Edge FunctionsのOpenAPI仕様書メンテナンス
- **Architecture Diagrams**: システム構成図の継続的更新
- **User Guides**: 機能変更時のユーザー向けガイド更新