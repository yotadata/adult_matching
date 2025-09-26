/**
 * ユーザー管理系Edge Functions共通認証ユーティリティ
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

// グローバル共有ユーティリティの使用
export {
  corsHeaders,
  optionsResponse,
  errorResponse,
  successResponse
} from "../../_shared/responses.ts";

export {
  getSupabaseClient,
  getSupabaseClientFromRequest,
  getSupabaseServiceClient
} from "../../_shared/database.ts";

export interface AuthResult {
  user_id: string;
  email?: string;
  authenticated: boolean;
  error?: string;
}

/**
 * リクエストからユーザー認証情報を取得
 */
export async function authenticateUser(req: Request): Promise<AuthResult> {
  try {
    const supabase = getSupabaseClient();
    
    // Authorizationヘッダーからトークンを取得
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return {
        user_id: '',
        authenticated: false,
        error: 'Authorization header missing'
      };
    }

    const token = authHeader.replace('Bearer ', '');
    
    // ユーザー認証を検証
    const { data: { user }, error } = await supabase.auth.getUser(token);
    
    if (error || !user) {
      console.error('Authentication error:', error);
      return {
        user_id: '',
        authenticated: false,
        error: error?.message || 'Invalid token'
      };
    }

    console.log(`Authenticated user: ${user.id}`);
    
    return {
      user_id: user.id,
      email: user.email,
      authenticated: true
    };

  } catch (error) {
    console.error('Authentication error:', error);
    return {
      user_id: '',
      authenticated: false,
      error: 'Authentication failed'
    };
  }
}

/**
 * 管理者権限チェック
 */
export async function checkAdminAccess(user_id: string): Promise<boolean> {
  try {
    const supabase = getSupabaseClient();
    
    // user_profilesテーブルで管理者権限を確認
    const { data, error } = await supabase
      .from('user_profiles')
      .select('is_admin')
      .eq('user_id', user_id)
      .single();

    if (error) {
      console.error('Admin check error:', error);
      return false;
    }

    return data?.is_admin === true;
  } catch (error) {
    console.error('Admin check error:', error);
    return false;
  }
}

/**
 * 認証が必要なエンドポイント用の標準エラーレスポンス
 */
export function unauthorizedResponse(message: string = 'Unauthorized') {
  return errorResponse(message, 401, 'UNAUTHORIZED');
}