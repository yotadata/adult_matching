/**
 * 統一データベース接続・操作ユーティリティ
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

export interface DatabaseConfig {
  useServiceRole?: boolean;
  authorizationHeader?: string;
}

/**
 * 標準Supabaseクライアントを取得（匿名キー使用）
 */
export function getSupabaseClient(config?: DatabaseConfig): SupabaseClient {
  const client = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_ANON_KEY') ?? '',
    config?.authorizationHeader ? {
      global: {
        headers: { Authorization: config.authorizationHeader },
      },
    } : undefined
  );

  return client;
}

/**
 * サービスロールSupabaseクライアントを取得（管理者権限）
 */
export function getSupabaseServiceClient(): SupabaseClient {
  return createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  );
}

/**
 * リクエストから適切なSupabaseクライアントを取得
 */
export function getSupabaseClientFromRequest(
  req: Request, 
  config?: DatabaseConfig
): SupabaseClient {
  if (config?.useServiceRole) {
    return getSupabaseServiceClient();
  }

  const authHeader = req.headers.get('Authorization');
  return getSupabaseClient({
    ...config,
    authorizationHeader: authHeader || undefined
  });
}

/**
 * データベース操作結果の共通インターフェース
 */
export interface DatabaseResult<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  count?: number;
}

/**
 * データベースエラーを統一形式に変換
 */
export function handleDatabaseError(error: any): DatabaseResult {
  console.error('Database operation error:', error);
  
  return {
    success: false,
    error: error?.message || 'Database operation failed'
  };
}

/**
 * データベース操作の成功結果を生成
 */
export function createSuccessResult<T>(data: T, count?: number): DatabaseResult<T> {
  return {
    success: true,
    data,
    count
  };
}

/**
 * ページネーション用のクエリビルダー
 */
export interface PaginationParams {
  limit?: number;
  offset?: number;
  page?: number;
}

export function applyPagination(
  query: any, 
  params: PaginationParams
): any {
  const limit = params.limit || 20;
  const offset = params.offset ?? (params.page ? (params.page - 1) * limit : 0);

  return query.range(offset, offset + limit - 1);
}

/**
 * 共通フィルタリング用のクエリビルダー
 */
export interface FilterParams {
  exclude_ids?: string[];
  [key: string]: any;
}

export function applyFilters(query: any, filters: FilterParams): any {
  if (filters.exclude_ids && filters.exclude_ids.length > 0) {
    query = query.not('id', 'in', filters.exclude_ids);
  }

  // 他の共通フィルタもここで処理
  return query;
}

/**
 * RPC関数の安全な呼び出し
 */
export async function callRPC<T>(
  client: SupabaseClient,
  functionName: string,
  params: any = {}
): Promise<DatabaseResult<T>> {
  try {
    const { data, error } = await client.rpc(functionName, params);
    
    if (error) {
      return handleDatabaseError(error);
    }

    return createSuccessResult(data);
  } catch (error) {
    return handleDatabaseError(error);
  }
}