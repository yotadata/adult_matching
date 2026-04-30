import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV,

  // 本番以外はサンプリングを下げる
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.2 : 0,
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: process.env.NODE_ENV === 'production' ? 1.0 : 0,

  integrations: [
    Sentry.replayIntegration({
      maskAllText: true,
      blockAllMedia: true,
    }),
  ],

  // DSN 未設定時は無効化
  enabled: !!process.env.NEXT_PUBLIC_SENTRY_DSN,
});
