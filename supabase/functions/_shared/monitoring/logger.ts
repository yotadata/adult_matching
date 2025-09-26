/**
 * Edge Functions Monitoring and Logging
 * 
 * Supabase Edge Functions用の統合ログとメトリクス
 * - 構造化ログ
 * - パフォーマンス監視
 * - エラー追跡
 * - リクエスト/レスポンス監視
 */

export interface LogContext {
  requestId?: string;
  userId?: string;
  functionName?: string;
  endpoint?: string;
  userAgent?: string;
  ipAddress?: string;
  timestamp?: string;
}

export interface MetricEvent {
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  tags?: Record<string, string>;
}

export interface ErrorEvent {
  error: Error | string;
  context: LogContext;
  stack?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export class EdgeFunctionLogger {
  private functionName: string;
  private baseContext: LogContext;

  constructor(functionName: string) {
    this.functionName = functionName;
    this.baseContext = {
      functionName,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * リクエストコンテキストの設定
   */
  setRequestContext(req: Request): LogContext {
    const url = new URL(req.url);
    const requestId = crypto.randomUUID();
    
    this.baseContext = {
      ...this.baseContext,
      requestId,
      endpoint: url.pathname,
      userAgent: req.headers.get('user-agent') || 'unknown',
      ipAddress: req.headers.get('x-forwarded-for') || 'unknown',
      timestamp: new Date().toISOString(),
    };

    return this.baseContext;
  }

  /**
   * ユーザーコンテキストの追加
   */
  setUserContext(userId: string): void {
    this.baseContext.userId = userId;
  }

  /**
   * 構造化情報ログ
   */
  info(message: string, data?: any, context?: Partial<LogContext>): void {
    const logEntry = {
      level: 'INFO',
      message,
      data,
      context: { ...this.baseContext, ...context },
      timestamp: new Date().toISOString(),
    };

    console.log(JSON.stringify(logEntry));
  }

  /**
   * 警告ログ
   */
  warn(message: string, data?: any, context?: Partial<LogContext>): void {
    const logEntry = {
      level: 'WARN',
      message,
      data,
      context: { ...this.baseContext, ...context },
      timestamp: new Date().toISOString(),
    };

    console.warn(JSON.stringify(logEntry));
  }

  /**
   * エラーログ
   */
  error(message: string, error?: Error | string, context?: Partial<LogContext>): void {
    const logEntry = {
      level: 'ERROR',
      message,
      error: error instanceof Error ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
      } : error,
      context: { ...this.baseContext, ...context },
      timestamp: new Date().toISOString(),
    };

    console.error(JSON.stringify(logEntry));
  }

  /**
   * デバッグログ（開発環境のみ）
   */
  debug(message: string, data?: any, context?: Partial<LogContext>): void {
    if (Deno.env.get('ENVIRONMENT') === 'development') {
      const logEntry = {
        level: 'DEBUG',
        message,
        data,
        context: { ...this.baseContext, ...context },
        timestamp: new Date().toISOString(),
      };

      console.debug(JSON.stringify(logEntry));
    }
  }

  /**
   * メトリクス記録
   */
  metric(name: string, value: number, unit: string = 'count', tags?: Record<string, string>): void {
    const metricEvent: MetricEvent = {
      name: `adult_matching_edge_${name}`,
      value,
      unit,
      timestamp: new Date().toISOString(),
      tags: {
        function: this.functionName,
        ...tags,
      },
    };

    this.info('METRIC', metricEvent);
  }

  /**
   * パフォーマンス計測開始
   */
  startTimer(operationName: string): () => void {
    const startTime = performance.now();
    
    return () => {
      const duration = performance.now() - startTime;
      this.metric(`${operationName}_duration_ms`, duration, 'milliseconds');
      this.debug(`Operation completed: ${operationName}`, { durationMs: duration });
    };
  }

  /**
   * リクエスト処理の監視
   */
  async monitorRequest<T>(
    req: Request,
    handler: (req: Request) => Promise<T>
  ): Promise<T> {
    const context = this.setRequestContext(req);
    const timer = this.startTimer('request_processing');
    
    this.info('Request started', {
      method: req.method,
      url: req.url,
    });

    try {
      const result = await handler(req);
      timer();
      
      this.info('Request completed successfully', {
        method: req.method,
        url: req.url,
      });

      this.metric('requests_total', 1, 'count', { status: 'success' });
      
      return result;
    } catch (error) {
      timer();
      
      this.error('Request failed', error as Error, context);
      this.metric('requests_total', 1, 'count', { status: 'error' });
      this.metric('errors_total', 1, 'count', { 
        error_type: error instanceof Error ? error.name : 'unknown' 
      });
      
      throw error;
    }
  }

  /**
   * データベース操作の監視
   */
  async monitorDatabaseOperation<T>(
    operationName: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const timer = this.startTimer(`db_${operationName}`);
    
    this.debug(`Database operation started: ${operationName}`);

    try {
      const result = await operation();
      timer();
      
      this.debug(`Database operation completed: ${operationName}`);
      this.metric('db_operations_total', 1, 'count', { 
        operation: operationName, 
        status: 'success' 
      });
      
      return result;
    } catch (error) {
      timer();
      
      this.error(`Database operation failed: ${operationName}`, error as Error);
      this.metric('db_operations_total', 1, 'count', { 
        operation: operationName, 
        status: 'error' 
      });
      this.metric('db_errors_total', 1, 'count', { 
        operation: operationName,
        error_type: error instanceof Error ? error.name : 'unknown'
      });
      
      throw error;
    }
  }

  /**
   * 機械学習操作の監視
   */
  async monitorMLOperation<T>(
    operationName: string,
    operation: () => Promise<T>,
    modelVersion?: string
  ): Promise<T> {
    const timer = this.startTimer(`ml_${operationName}`);
    
    this.info(`ML operation started: ${operationName}`, {
      modelVersion,
    });

    try {
      const result = await operation();
      timer();
      
      this.info(`ML operation completed: ${operationName}`, {
        modelVersion,
      });
      
      this.metric('ml_operations_total', 1, 'count', { 
        operation: operationName, 
        status: 'success',
        model_version: modelVersion || 'unknown'
      });
      
      return result;
    } catch (error) {
      timer();
      
      this.error(`ML operation failed: ${operationName}`, error as Error);
      this.metric('ml_operations_total', 1, 'count', { 
        operation: operationName, 
        status: 'error',
        model_version: modelVersion || 'unknown'
      });
      this.metric('ml_errors_total', 1, 'count', { 
        operation: operationName,
        error_type: error instanceof Error ? error.name : 'unknown'
      });
      
      throw error;
    }
  }

  /**
   * ヘルスチェック情報の記録
   */
  recordHealthCheck(status: 'healthy' | 'degraded' | 'unhealthy', details?: any): void {
    this.info('Health check performed', {
      status,
      details,
    });

    this.metric('health_checks_total', 1, 'count', { status });
  }

  /**
   * ユーザーアクションの記録
   */
  recordUserAction(action: string, userId: string, metadata?: any): void {
    this.info('User action recorded', {
      action,
      userId,
      metadata,
    });

    this.metric('user_actions_total', 1, 'count', { action });
  }

  /**
   * レスポンス送信の監視
   */
  monitorResponse(response: Response, context?: Partial<LogContext>): Response {
    this.info('Response sent', {
      status: response.status,
      statusText: response.statusText,
    }, context);

    this.metric('responses_total', 1, 'count', { 
      status: response.status.toString(),
      status_class: `${Math.floor(response.status / 100)}xx`
    });

    return response;
  }
}

/**
 * レスポンスヘルパー関数
 */
export class ResponseHelper {
  private logger: EdgeFunctionLogger;

  constructor(logger: EdgeFunctionLogger) {
    this.logger = logger;
  }

  /**
   * 成功レスポンス
   */
  success(data: any, status: number = 200): Response {
    const response = new Response(JSON.stringify({
      success: true,
      data,
      timestamp: new Date().toISOString(),
    }), {
      status,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
      },
    });

    return this.logger.monitorResponse(response);
  }

  /**
   * エラーレスポンス
   */
  error(message: string, status: number = 500, errorCode?: string): Response {
    this.logger.error('Error response sent', message);

    const response = new Response(JSON.stringify({
      success: false,
      error: {
        message,
        code: errorCode,
        timestamp: new Date().toISOString(),
      },
    }), {
      status,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
      },
    });

    return this.logger.monitorResponse(response);
  }

  /**
   * バリデーションエラーレスポンス
   */
  validationError(errors: string[]): Response {
    this.logger.warn('Validation error', { errors });

    const response = new Response(JSON.stringify({
      success: false,
      error: {
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        details: errors,
        timestamp: new Date().toISOString(),
      },
    }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
      },
    });

    return this.logger.monitorResponse(response);
  }

  /**
   * 認証エラーレスポンス
   */
  unauthorized(message: string = 'Unauthorized'): Response {
    this.logger.warn('Unauthorized access attempt');

    const response = new Response(JSON.stringify({
      success: false,
      error: {
        message,
        code: 'UNAUTHORIZED',
        timestamp: new Date().toISOString(),
      },
    }), {
      status: 401,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
      },
    });

    return this.logger.monitorResponse(response);
  }

  /**
   * レート制限エラーレスポンス
   */
  rateLimit(retryAfter?: number): Response {
    this.logger.warn('Rate limit exceeded');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-cache',
    };

    if (retryAfter) {
      headers['Retry-After'] = retryAfter.toString();
    }

    const response = new Response(JSON.stringify({
      success: false,
      error: {
        message: 'Rate limit exceeded',
        code: 'RATE_LIMIT_EXCEEDED',
        retryAfter,
        timestamp: new Date().toISOString(),
      },
    }), {
      status: 429,
      headers,
    });

    return this.logger.monitorResponse(response);
  }
}

/**
 * ファクトリー関数
 */
export function createLogger(functionName: string): EdgeFunctionLogger {
  return new EdgeFunctionLogger(functionName);
}

export function createResponseHelper(logger: EdgeFunctionLogger): ResponseHelper {
  return new ResponseHelper(logger);
}

/**
 * デコレーター関数 - Edge Function のメイン処理をラップ
 */
export function withMonitoring(functionName: string) {
  return function(handler: (req: Request) => Promise<Response>) {
    return async function(req: Request): Promise<Response> {
      const logger = createLogger(functionName);
      const responseHelper = createResponseHelper(logger);

      try {
        return await logger.monitorRequest(req, handler);
      } catch (error) {
        logger.error('Unhandled error in Edge Function', error as Error);
        return responseHelper.error('Internal server error', 500, 'INTERNAL_ERROR');
      }
    };
  };
}

/**
 * ヘルスチェックエンドポイント用ヘルパー
 */
export function createHealthCheck(
  functionName: string,
  checkFunctions: Array<() => Promise<{ status: string; details?: any }>>
) {
  return async function(req: Request): Promise<Response> {
    const logger = createLogger(functionName);
    const responseHelper = createResponseHelper(logger);

    try {
      const results = await Promise.all(
        checkFunctions.map(async (check, index) => {
          try {
            const result = await check();
            return { check: index, ...result };
          } catch (error) {
            return { 
              check: index, 
              status: 'unhealthy', 
              error: error instanceof Error ? error.message : 'Unknown error' 
            };
          }
        })
      );

      const overallStatus = results.every(r => r.status === 'healthy') 
        ? 'healthy' 
        : results.some(r => r.status === 'unhealthy') 
        ? 'unhealthy' 
        : 'degraded';

      logger.recordHealthCheck(overallStatus, results);

      const statusCode = overallStatus === 'healthy' ? 200 : 
                        overallStatus === 'degraded' ? 200 : 503;

      return responseHelper.success({
        status: overallStatus,
        checks: results,
        timestamp: new Date().toISOString(),
      }, statusCode);

    } catch (error) {
      logger.error('Health check failed', error as Error);
      return responseHelper.error('Health check failed', 503, 'HEALTH_CHECK_FAILED');
    }
  };
}