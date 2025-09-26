/**
 * 統一HTTPレスポンス処理ユーティリティ
 */

export const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
};

/**
 * API応答の共通インターフェース
 */
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  code?: string;
  meta?: {
    processing_time_ms?: number;
    pagination?: PaginationMeta;
    performance?: any;
  };
}

export interface PaginationMeta {
  limit: number;
  offset: number;
  total_count?: number;
  has_more: boolean;
}

/**
 * レスポンスビルダークラス
 */
export class ResponseBuilder<T = any> {
  private response: ApiResponse<T> = { success: true };

  /**
   * 成功データを設定
   */
  data(data: T): this {
    this.response.success = true;
    this.response.data = data;
    return this;
  }

  /**
   * エラーを設定
   */
  error(message: string, code?: string): this {
    this.response.success = false;
    this.response.error = message;
    if (code) {
      this.response.code = code;
    }
    return this;
  }

  /**
   * メッセージを設定
   */
  message(message: string): this {
    this.response.message = message;
    return this;
  }

  /**
   * 処理時間を設定
   */
  processingTime(timeMs: number): this {
    if (!this.response.meta) {
      this.response.meta = {};
    }
    this.response.meta.processing_time_ms = timeMs;
    return this;
  }

  /**
   * ページネーション情報を設定
   */
  pagination(meta: PaginationMeta): this {
    if (!this.response.meta) {
      this.response.meta = {};
    }
    this.response.meta.pagination = meta;
    return this;
  }

  /**
   * パフォーマンス情報を設定
   */
  performance(metrics: any): this {
    if (!this.response.meta) {
      this.response.meta = {};
    }
    this.response.meta.performance = metrics;
    return this;
  }

  /**
   * HTTPレスポンスを構築
   */
  build(status?: number): Response {
    const httpStatus = status || (this.response.success ? 200 : 400);
    
    return new Response(
      JSON.stringify(this.response),
      {
        status: httpStatus,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      }
    );
  }
}

/**
 * 成功レスポンスの作成
 */
export function successResponse<T>(
  data: T,
  message?: string,
  processingTimeMs?: number
): Response {
  const builder = new ResponseBuilder<T>().data(data);
  
  if (message) {
    builder.message(message);
  }
  
  if (processingTimeMs !== undefined) {
    builder.processingTime(processingTimeMs);
  }
  
  return builder.build();
}

/**
 * エラーレスポンスの作成
 */
export function errorResponse(
  message: string,
  status: number = 400,
  code?: string,
  processingTimeMs?: number
): Response {
  const builder = new ResponseBuilder().error(message, code);
  
  if (processingTimeMs !== undefined) {
    builder.processingTime(processingTimeMs);
  }
  
  return builder.build(status);
}

/**
 * 認証エラーレスポンス
 */
export function unauthorizedResponse(
  message: string = 'Unauthorized',
  processingTimeMs?: number
): Response {
  return errorResponse(message, 401, 'UNAUTHORIZED', processingTimeMs);
}

/**
 * 禁止アクセスレスポンス
 */
export function forbiddenResponse(
  message: string = 'Forbidden',
  processingTimeMs?: number
): Response {
  return errorResponse(message, 403, 'FORBIDDEN', processingTimeMs);
}

/**
 * 見つからないレスポンス
 */
export function notFoundResponse(
  message: string = 'Not Found',
  processingTimeMs?: number
): Response {
  return errorResponse(message, 404, 'NOT_FOUND', processingTimeMs);
}

/**
 * バリデーションエラーレスポンス
 */
export function validationErrorResponse(
  errors: string[],
  processingTimeMs?: number
): Response {
  return errorResponse(
    `Validation failed: ${errors.join(', ')}`,
    422,
    'VALIDATION_ERROR',
    processingTimeMs
  );
}

/**
 * サーバーエラーレスポンス
 */
export function internalErrorResponse(
  message: string = 'Internal Server Error',
  processingTimeMs?: number
): Response {
  return errorResponse(message, 500, 'INTERNAL_ERROR', processingTimeMs);
}

/**
 * OPTIONSリクエスト用レスポンス
 */
export function optionsResponse(): Response {
  return new Response('ok', { headers: corsHeaders });
}

/**
 * ページネーション付き成功レスポンス
 */
export function paginatedResponse<T>(
  data: T[],
  pagination: PaginationMeta,
  message?: string,
  processingTimeMs?: number
): Response {
  const builder = new ResponseBuilder<T[]>()
    .data(data)
    .pagination(pagination);
  
  if (message) {
    builder.message(message);
  }
  
  if (processingTimeMs !== undefined) {
    builder.processingTime(processingTimeMs);
  }
  
  return builder.build();
}

/**
 * パフォーマンス情報付きレスポンス
 */
export function performanceResponse<T>(
  data: T,
  performanceMetrics: any,
  message?: string
): Response {
  return new ResponseBuilder<T>()
    .data(data)
    .performance(performanceMetrics)
    .message(message)
    .processingTime(performanceMetrics.totalDurationMs)
    .build();
}

/**
 * ストリーミングレスポンス（将来的な拡張用）
 */
export function streamingResponse(
  stream: ReadableStream,
  contentType: string = 'application/json'
): Response {
  return new Response(stream, {
    headers: {
      ...corsHeaders,
      'Content-Type': contentType,
      'Transfer-Encoding': 'chunked',
    },
  });
}

/**
 * レスポンスヘルパー関数群
 */
export const responses = {
  success: successResponse,
  error: errorResponse,
  unauthorized: unauthorizedResponse,
  forbidden: forbiddenResponse,
  notFound: notFoundResponse,
  validationError: validationErrorResponse,
  internalError: internalErrorResponse,
  options: optionsResponse,
  paginated: paginatedResponse,
  performance: performanceResponse,
  streaming: streamingResponse,
  builder: <T = any>() => new ResponseBuilder<T>()
};