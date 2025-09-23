/**
 * ユーザー管理系Edge Functions共通データベースユーティリティ
 */

import { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

// Inline definitions to avoid import issues
export interface DatabaseResult<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  count?: number;
}

export function createSuccessResult<T>(data: T, count?: number): DatabaseResult<T> {
  return {
    success: true,
    data,
    count
  };
}

export function handleDatabaseError(error: any): DatabaseResult {
  console.error('Database operation error:', error);

  return {
    success: false,
    error: error?.message || 'Database operation failed'
  };
}

export interface PaginationParams {
  limit?: number;
  offset?: number;
  page?: number;
}

export async function callRPC<T>(
  client: any,
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

export interface DatabaseError {
  code: string;
  message: string;
  details?: any;
}

/**
 * ユーザーのいいね数を取得
 */
export async function getUserLikeCount(
  supabase: SupabaseClient, 
  user_id: string
): Promise<number> {
  try {
    const { count, error } = await supabase
      .from('likes')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', user_id);

    if (error) {
      console.error('Error getting like count:', error);
      return 0;
    }

    return count || 0;
  } catch (error) {
    console.error('Error getting like count:', error);
    return 0;
  }
}

/**
 * ユーザーの埋め込みが存在するかチェック
 */
export async function checkUserEmbeddingExists(
  supabase: SupabaseClient, 
  user_id: string
): Promise<boolean> {
  try {
    const { data, error } = await supabase
      .from('user_embeddings')
      .select('user_id')
      .eq('user_id', user_id)
      .single();

    return !error && !!data;
  } catch (error) {
    return false;
  }
}

/**
 * ユーザーの関連データをクリーンアップ
 */
export async function cleanupUserData(
  supabase: SupabaseClient, 
  user_id: string
): Promise<{ success: boolean; error?: DatabaseError }> {
  try {
    // トランザクション的に複数のテーブルから削除
    const deleteTables = [
      'likes',
      'user_embeddings', 
      'user_preferences',
      'user_behavior_log'
    ];

    for (const table of deleteTables) {
      const { error } = await supabase
        .from(table)
        .delete()
        .eq('user_id', user_id);

      if (error) {
        console.error(`Error deleting from ${table}:`, error);
        return {
          success: false,
          error: {
            code: 'CLEANUP_ERROR',
            message: `Failed to cleanup ${table}`,
            details: error
          }
        };
      }
    }

    console.log(`Successfully cleaned up data for user: ${user_id}`);
    return { success: true };

  } catch (error) {
    console.error('Cleanup error:', error);
    return {
      success: false,
      error: {
        code: 'CLEANUP_EXCEPTION',
        message: 'Unexpected error during cleanup',
        details: error
      }
    };
  }
}

/**
 * いいねの重複チェック
 */
export async function checkDuplicateLike(
  supabase: SupabaseClient,
  user_id: string,
  video_id: string
): Promise<boolean> {
  try {
    const { data, error } = await supabase
      .from('likes')
      .select('id')
      .eq('user_id', user_id)
      .eq('video_id', video_id)
      .single();

    return !error && !!data;
  } catch (error) {
    return false;
  }
}

/**
 * ユーザーの最近のアクティビティを取得
 */
export async function getUserRecentActivity(
  supabase: SupabaseClient,
  user_id: string,
  limit: number = 10
): Promise<any[]> {
  try {
    const { data, error } = await supabase
      .from('likes')
      .select(`
        video_id,
        created_at,
        videos:video_id (
          title,
          maker,
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', user_id)
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      console.error('Error getting user activity:', error);
      return [];
    }

    return data || [];
  } catch (error) {
    console.error('Error getting user activity:', error);
    return [];
  }
}

/**
 * ユーザーのジャンル嗜好を分析
 */
export async function analyzeUserGenrePreferences(
  supabase: SupabaseClient,
  user_id: string
): Promise<{ [genre: string]: number }> {
  try {
    const { data, error } = await supabase
      .from('likes')
      .select(`
        videos:video_id (
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', user_id);

    if (error || !data) {
      console.error('Error analyzing genre preferences:', error);
      return {};
    }

    // ジャンル別の集計
    const genreCount: { [genre: string]: number } = {};
    
    data.forEach((like: any) => {
      if (like.videos?.video_tags) {
        const tags = like.videos.video_tags.map((vt: any) => vt.tags?.name).filter(Boolean);
        const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));
        if (genreTag) {
          genreCount[genreTag] = (genreCount[genreTag] || 0) + 1;
        }
      }
    });

    return genreCount;
  } catch (error) {
    console.error('Error analyzing genre preferences:', error);
    return {};
  }
}

/**
 * データベース操作の実行時間を測定
 */
export async function measureDbOperation<T>(
  operation: () => Promise<T>,
  operationName: string
): Promise<{ result: T; duration: number }> {
  const startTime = performance.now();
  
  try {
    const result = await operation();
    const duration = performance.now() - startTime;
    
    console.log(`${operationName} completed in ${duration.toFixed(2)}ms`);
    
    return { result, duration };
  } catch (error) {
    const duration = performance.now() - startTime;
    console.error(`${operationName} failed after ${duration.toFixed(2)}ms:`, error);
    throw error;
  }
}