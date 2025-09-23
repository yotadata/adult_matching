# Code Style and Conventions

## TypeScript Conventions
- **Strict Mode**: TypeScript strict mode enabled
- **Interface Naming**: PascalCase (例: `CardData`, `SwipeCardProps`)
- **Type Exports**: Export interfaces and types alongside components
- **Optional Properties**: Use `?` for optional properties
- **Function Types**: Use arrow function types for callbacks

## React Conventions
- **Functional Components**: All components use function syntax
- **Hooks**: Extensive use of React hooks (useState, useEffect, useRef)
- **Forward Refs**: Use `forwardRef` for imperative handles
- **Custom Hooks**: Custom hooks start with `use` (例: `useMediaQuery`)
- **Client Components**: Use `'use client'` directive at top of files

## File Structure
- **Component Files**: `.tsx` extension for React components
- **API Routes**: `/api/` directory with `route.ts` files
- **Hooks**: `/hooks/` directory
- **Lib**: `/lib/` directory for utilities
- **Absolute Imports**: Use `@/*` path mapping

## Naming Conventions
- **Components**: PascalCase (例: `SwipeCard`, `MobileVideoLayout`)
- **Files**: kebab-case for component files (例: `SwipeCard.tsx`)
- **Functions**: camelCase (例: `handleSwipe`, `fetchVideos`)
- **Constants**: UPPER_SNAKE_CASE (例: `ORIGINAL_GRADIENT`)
- **Props Interfaces**: ComponentName + "Props" (例: `SwipeCardProps`)

## Comments
- **Japanese Comments**: コメントは日本語で記述
- **Type Annotations**: インターフェースの説明コメント
- **Function Documentation**: 複雑な関数には動作説明コメント

## ESLint Configuration
- Extends `next/core-web-vitals` and `next/typescript`
- Custom rule: `react/jsx-no-duplicate-props` disabled

## Code Organization
- **Separation of Concerns**: UI、ロジック、型定義を明確に分離
- **Single Responsibility**: コンポーネントは単一責任の原則
- **Reusability**: 再利用可能なコンポーネント設計
- **Error Boundaries**: エラーハンドリングの適切な実装

## Edge Functions (Supabase)
- **TypeScript**: Deno環境でTypeScript使用
- **CORS Headers**: 全てのAPIレスポンスにCORSヘッダー
- **Error Handling**: 統一されたエラーレスポンス形式
- **Authentication**: JWT認証の一貫した実装