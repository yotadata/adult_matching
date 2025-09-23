/**
 * 統一認証ユーティリティ
 * Authentication utilities for Edge Functions
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import type { SupabaseClient, User } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import { getSupabaseClient, getSupabaseClientFromRequest } from "./database.ts";
import { corsHeaders, errorResponse, unauthorizedResponse } from "./responses.ts";
import type { AuthResult } from "./types.ts";

/**
 * リクエストからユーザー認証を行う統一関数
 * Unified user authentication from request
 */
export async function authenticateUser(req: Request): Promise<AuthResult> {
  try {
    const authHeader = req.headers.get('Authorization');

    if (!authHeader) {
      return {
        user_id: '',
        authenticated: false,
        error: 'No authorization header'
      };
    }

    const supabaseClient = getSupabaseClient({
      authorizationHeader: authHeader
    });

    const { data: { user }, error } = await supabaseClient.auth.getUser();

    if (error || !user) {
      return {
        user_id: '',
        authenticated: false,
        error: error?.message || 'Authentication failed'
      };
    }

    return {
      user_id: user.id,
      email: user.email,
      authenticated: true
    };

  } catch (error) {
    return {
      user_id: '',
      authenticated: false,
      error: `Authentication error: ${error.message}`
    };
  }
}

/**
 * 認証が必要なエンドポイント用のミドルウェア
 * Middleware for endpoints requiring authentication
 */
export async function requireAuth(
  req: Request,
  handler: (req: Request, user: AuthResult) => Promise<Response>
): Promise<Response> {
  const authResult = await authenticateUser(req);

  if (!authResult.authenticated) {
    return unauthorizedResponse(authResult.error || 'Authentication required');
  }

  return handler(req, authResult);
}

/**
 * オプション認証（認証があればユーザー情報を提供、なくても処理続行）
 * Optional authentication (provides user info if authenticated, continues if not)
 */
export async function optionalAuth(req: Request): Promise<AuthResult> {
  const authHeader = req.headers.get('Authorization');

  if (!authHeader) {
    return {
      user_id: '',
      authenticated: false
    };
  }

  return await authenticateUser(req);
}

/**
 * サービスロール認証チェック
 * Service role authentication check
 */
export function isServiceRole(req: Request): boolean {
  const authHeader = req.headers.get('Authorization');
  const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

  if (!authHeader || !serviceKey) {
    return false;
  }

  return authHeader.includes(serviceKey);
}

/**
 * 管理者権限チェック
 * Admin permission check
 */
export async function requireAdmin(
  req: Request,
  handler: (req: Request, user: AuthResult) => Promise<Response>
): Promise<Response> {
  // Service role has admin privileges
  if (isServiceRole(req)) {
    return handler(req, {
      user_id: 'service_role',
      authenticated: true
    });
  }

  const authResult = await authenticateUser(req);

  if (!authResult.authenticated) {
    return unauthorizedResponse('Authentication required');
  }

  // Check if user has admin role in profiles table
  const supabaseClient = getSupabaseClientFromRequest(req);
  const { data: profile, error } = await supabaseClient
    .from('profiles')
    .select('is_admin')
    .eq('user_id', authResult.user_id)
    .single();

  if (error || !profile?.is_admin) {
    return errorResponse('Admin access required', 403);
  }

  return handler(req, authResult);
}

/**
 * レート制限チェック（簡易版）
 * Simple rate limiting check
 */
export class RateLimiter {
  private static requests = new Map<string, { count: number; resetTime: number }>();

  static check(
    identifier: string,
    limit: number = 100,
    windowMs: number = 60000
  ): boolean {
    const now = Date.now();
    const key = identifier;
    const existing = this.requests.get(key);

    if (!existing || now > existing.resetTime) {
      this.requests.set(key, { count: 1, resetTime: now + windowMs });
      return true;
    }

    if (existing.count >= limit) {
      return false;
    }

    existing.count++;
    return true;
  }

  static getRemainingRequests(
    identifier: string,
    limit: number = 100
  ): number {
    const existing = this.requests.get(identifier);
    if (!existing) return limit;
    return Math.max(0, limit - existing.count);
  }
}

/**
 * レート制限ミドルウェア
 * Rate limiting middleware
 */
export async function withRateLimit(
  req: Request,
  handler: (req: Request) => Promise<Response>,
  options: {
    authenticated_limit?: number;
    anonymous_limit?: number;
    window_ms?: number;
  } = {}
): Promise<Response> {
  const {
    authenticated_limit = 1000,
    anonymous_limit = 100,
    window_ms = 60000
  } = options;

  const authResult = await optionalAuth(req);
  const identifier = authResult.authenticated ?
    `user:${authResult.user_id}` :
    `ip:${req.headers.get('x-forwarded-for') || 'unknown'}`;

  const limit = authResult.authenticated ? authenticated_limit : anonymous_limit;

  if (!RateLimiter.check(identifier, limit, window_ms)) {
    return new Response(
      JSON.stringify({
        error: 'Rate limit exceeded',
        retry_after: Math.ceil(window_ms / 1000)
      }),
      {
        status: 429,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'Retry-After': Math.ceil(window_ms / 1000).toString(),
          'X-RateLimit-Limit': limit.toString(),
          'X-RateLimit-Remaining': RateLimiter.getRemainingRequests(identifier, limit).toString()
        }
      }
    );
  }

  const response = await handler(req);

  // Add rate limit headers to response
  const remaining = RateLimiter.getRemainingRequests(identifier, limit);
  response.headers.set('X-RateLimit-Limit', limit.toString());
  response.headers.set('X-RateLimit-Remaining', remaining.toString());

  return response;
}

/**
 * JWT トークン検証（追加セキュリティ）
 * JWT token validation (additional security)
 */
export async function validateJWT(token: string): Promise<User | null> {
  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    );

    const { data: { user }, error } = await supabaseClient.auth.getUser(token.replace('Bearer ', ''));

    if (error || !user) {
      return null;
    }

    return user;
  } catch (error) {
    console.error('JWT validation error:', error);
    return null;
  }
}

/**
 * セッション情報の取得
 * Get session information
 */
export async function getSession(req: Request): Promise<{
  user: User | null;
  session: any;
  error?: string;
}> {
  try {
    const supabaseClient = getSupabaseClientFromRequest(req);
    const { data, error } = await supabaseClient.auth.getSession();

    if (error) {
      return { user: null, session: null, error: error.message };
    }

    return {
      user: data.session?.user || null,
      session: data.session
    };
  } catch (error) {
    return {
      user: null,
      session: null,
      error: `Session error: ${error.message}`
    };
  }
}

// Export legacy functions for backward compatibility
export { corsHeaders, errorResponse, unauthorizedResponse } from "./responses.ts";
export { getSupabaseClient, getSupabaseClientFromRequest, getSupabaseServiceClient } from "./database.ts";