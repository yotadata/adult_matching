/**
 * 共有ユーティリティ統一エクスポート
 */

// データベース関連
export {
  getSupabaseClient,
  getSupabaseServiceClient,
  getSupabaseClientFromRequest,
  handleDatabaseError,
  createSuccessResult,
  applyPagination,
  applyFilters,
  callRPC,
  type DatabaseConfig,
  type DatabaseResult,
  type PaginationParams,
  type FilterParams
} from './database.ts';

// レスポンス関連
export {
  corsHeaders,
  ResponseBuilder,
  successResponse,
  errorResponse,
  unauthorizedResponse,
  forbiddenResponse,
  notFoundResponse,
  validationErrorResponse,
  internalErrorResponse,
  optionsResponse,
  paginatedResponse,
  performanceResponse,
  streamingResponse,
  responses,
  type ApiResponse,
  type PaginationMeta
} from './responses.ts';

// バリデーション関連
export {
  Validator,
  validatePagination,
  validateFeedRequest,
  validateRecommendationRequest,
  parseAndValidate,
  type ValidationResult,
  type PaginationValidation,
  type FeedValidation,
  type RecommendationValidation
} from './validation.ts';

// モニタリング関連
export {
  LogLevel,
  Logger,
  createLogger,
  PerformanceMonitor,
  ErrorTracker,
  performHealthCheck,
  type LogContext,
  type PerformanceMetrics,
  type ErrorContext,
  type HealthMetrics
} from './monitoring.ts';

// 既存のユーティリティ（後方互換性）
export {
  getModelLoader,
  globalModelLoader,
  initializeModelLoader,
  TensorFlowJSModelLoader,
  type ModelLoadResult,
  type ModelConfig,
  type ModelCache,
  type InferenceResult
} from './model_loader.ts';

export {
  FeaturePreprocessor,
  getFeaturePreprocessor,
  globalPreprocessor,
  initializeFeaturePreprocessor,
  type UserFeatures,
  type ItemFeatures,
  type TensorFlowJSInput,
  type ItemTensorFlowJSInput,
  type ProcessingStats
} from './feature_preprocessor.ts';

// 型定義の再エクスポート
export type * from './types.ts';

/**
 * 便利なファクトリー関数
 */
export interface EdgeFunctionContext {
  functionName: string;
  req: Request;
  startTime: number;
  logger: Logger;
  monitor: PerformanceMonitor;
  errorTracker: ErrorTracker;
}

/**
 * Edge Function用の標準セットアップ
 */
export function createFunctionContext(functionName: string, req: Request): EdgeFunctionContext {
  const startTime = performance.now();
  const logger = createLogger(functionName, req);
  const monitor = new PerformanceMonitor(functionName, logger);
  const errorTracker = new ErrorTracker(logger);

  logger.info(`Function started: ${functionName}`, {
    method: req.method,
    url: req.url
  });

  return {
    functionName,
    req,
    startTime,
    logger,
    monitor,
    errorTracker
  };
}

/**
 * 標準的なOPTIONSハンドラー
 */
export function handleOptions(): Response {
  return optionsResponse();
}

/**
 * 標準的なメソッドチェック
 */
export function validateMethod(req: Request, allowedMethods: string[]): boolean {
  return allowedMethods.includes(req.method);
}

/**
 * エラーハンドリング付きの関数実行ラッパー
 */
export async function withErrorHandling<T>(
  context: EdgeFunctionContext,
  operation: string,
  fn: () => Promise<T>
): Promise<T> {
  const timer = context.logger.startTimer(operation);
  
  try {
    const result = await fn();
    const duration = timer();
    context.monitor.recordOperation(operation, duration);
    return result;
  } catch (error) {
    timer();
    context.errorTracker.trackError(operation, error);
    throw error;
  }
}

/**
 * 標準的なリクエスト終了処理
 */
export function finalizeFunctionExecution(
  context: EdgeFunctionContext,
  response?: Response
): Response {
  const metrics = context.monitor.getMetrics();
  
  if (response) {
    context.logger.info(`Function completed successfully: ${context.functionName}`, {
      status: response.status,
      metrics
    });
    return response;
  }

  // デフォルトの成功レスポンス
  return performanceResponse(
    { success: true },
    metrics,
    `Function ${context.functionName} completed successfully`
  );
}