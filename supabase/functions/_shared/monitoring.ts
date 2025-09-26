/**
 * 統一監視・ログ・パフォーマンス計測ユーティリティ
 */

export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR'
}

export interface LogContext {
  functionName: string;
  userId?: string;
  requestId?: string;
  correlationId?: string;
  [key: string]: any;
}

/**
 * 統一ログ出力クラス
 */
export class Logger {
  private context: LogContext;

  constructor(context: LogContext) {
    this.context = context;
  }

  private formatMessage(level: LogLevel, message: string, data?: any): string {
    const timestamp = new Date().toISOString();
    const contextStr = JSON.stringify(this.context);
    const dataStr = data ? ` | Data: ${JSON.stringify(data)}` : '';
    
    return `[${timestamp}] [${level}] [${this.context.functionName}] ${message} | Context: ${contextStr}${dataStr}`;
  }

  debug(message: string, data?: any): void {
    console.debug(this.formatMessage(LogLevel.DEBUG, message, data));
  }

  info(message: string, data?: any): void {
    console.info(this.formatMessage(LogLevel.INFO, message, data));
  }

  warn(message: string, data?: any): void {
    console.warn(this.formatMessage(LogLevel.WARN, message, data));
  }

  error(message: string, error?: any): void {
    const errorData = error instanceof Error ? {
      message: error.message,
      stack: error.stack
    } : error;
    
    console.error(this.formatMessage(LogLevel.ERROR, message, errorData));
  }

  /**
   * パフォーマンス測定開始
   */
  startTimer(operation: string): () => number {
    const startTime = performance.now();
    this.debug(`Starting operation: ${operation}`);
    
    return () => {
      const duration = performance.now() - startTime;
      this.info(`Operation completed: ${operation}`, { durationMs: duration });
      return duration;
    };
  }

  /**
   * コンテキスト情報を追加
   */
  addContext(additionalContext: Record<string, any>): Logger {
    return new Logger({
      ...this.context,
      ...additionalContext
    });
  }
}

/**
 * ロガーファクトリー
 */
export function createLogger(functionName: string, req?: Request): Logger {
  const requestId = crypto.randomUUID();
  const context: LogContext = {
    functionName,
    requestId
  };

  // リクエストからユーザーIDを抽出（可能であれば）
  if (req) {
    const authHeader = req.headers.get('Authorization');
    if (authHeader) {
      // Bearerトークンからユーザー情報を取得する処理は省略
      // 実際のプロジェクトでは適切に実装する
    }
  }

  return new Logger(context);
}

/**
 * パフォーマンス計測データ
 */
export interface PerformanceMetrics {
  functionName: string;
  totalDurationMs: number;
  operationMetrics: Record<string, number>;
  memoryUsage?: {
    used: number;
    total: number;
  };
  timestamp: string;
}

/**
 * パフォーマンスモニタリングクラス
 */
export class PerformanceMonitor {
  private functionName: string;
  private startTime: number;
  private operations: Record<string, number> = {};
  private logger: Logger;

  constructor(functionName: string, logger: Logger) {
    this.functionName = functionName;
    this.startTime = performance.now();
    this.logger = logger;
  }

  /**
   * 操作時間を記録
   */
  recordOperation(name: string, duration: number): void {
    this.operations[name] = duration;
    this.logger.debug(`Operation recorded: ${name}`, { durationMs: duration });
  }

  /**
   * メモリ使用量を取得（可能であれば）
   */
  private getMemoryUsage() {
    try {
      // Deno環境でメモリ使用量を取得
      return {
        used: (performance as any).memory?.usedJSHeapSize || 0,
        total: (performance as any).memory?.totalJSHeapSize || 0
      };
    } catch {
      return undefined;
    }
  }

  /**
   * 最終パフォーマンス結果を生成
   */
  getMetrics(): PerformanceMetrics {
    const totalDuration = performance.now() - this.startTime;
    
    const metrics: PerformanceMetrics = {
      functionName: this.functionName,
      totalDurationMs: totalDuration,
      operationMetrics: { ...this.operations },
      memoryUsage: this.getMemoryUsage(),
      timestamp: new Date().toISOString()
    };

    this.logger.info('Performance metrics collected', metrics);
    return metrics;
  }
}

/**
 * エラー追跡とレポーティング
 */
export interface ErrorContext {
  functionName: string;
  operation: string;
  userId?: string;
  requestData?: any;
  error: Error | string;
  timestamp: string;
}

export class ErrorTracker {
  private logger: Logger;

  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * エラーを追跡し、適切にログ出力
   */
  trackError(
    operation: string,
    error: Error | string,
    context?: Record<string, any>
  ): ErrorContext {
    const errorContext: ErrorContext = {
      functionName: this.logger['context'].functionName,
      operation,
      userId: this.logger['context'].userId,
      requestData: context?.requestData,
      error: error instanceof Error ? error.message : error,
      timestamp: new Date().toISOString()
    };

    this.logger.error(`Error in operation: ${operation}`, {
      ...errorContext,
      stack: error instanceof Error ? error.stack : undefined,
      ...context
    });

    return errorContext;
  }

  /**
   * 重要なエラーをアラート（将来的に外部サービス連携）
   */
  alertCriticalError(errorContext: ErrorContext): void {
    // 将来的にSlack、Discord、またはアラートサービスと連携
    this.logger.error('CRITICAL ERROR DETECTED', errorContext);
  }
}

/**
 * 健全性チェック用のメトリクス
 */
export interface HealthMetrics {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: {
    database: boolean;
    storage: boolean;
    ai_models: boolean;
  };
  responseTimeMs: number;
  timestamp: string;
}

/**
 * 健全性チェック実行
 */
export async function performHealthCheck(logger: Logger): Promise<HealthMetrics> {
  const startTime = performance.now();
  
  const checks = {
    database: true, // 実際にはデータベース接続をテスト
    storage: true,  // 実際にはSupabase Storageをテスト
    ai_models: true // 実際にはモデルロードをテスト
  };

  const responseTime = performance.now() - startTime;
  const failedChecks = Object.values(checks).filter(check => !check).length;
  
  let status: HealthMetrics['status'] = 'healthy';
  if (failedChecks > 0) {
    status = failedChecks === Object.keys(checks).length ? 'unhealthy' : 'degraded';
  }

  const metrics: HealthMetrics = {
    status,
    checks,
    responseTimeMs: responseTime,
    timestamp: new Date().toISOString()
  };

  logger.info('Health check completed', metrics);
  return metrics;
}