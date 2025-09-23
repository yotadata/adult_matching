import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { 
  corsHeaders, 
  optionsResponse,
  errorResponse
} from "./_shared/auth.ts";

/**
 * ユーザー管理関数グループのメインルーター
 * 
 * パス構造:
 * - /user-management/likes → いいね管理
 * - /user-management/embeddings → エンベディング更新
 * - /user-management/profile → プロフィール管理
 * - /user-management/account → アカウント削除
 */

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  try {
    const url = new URL(req.url);
    const path = url.pathname.replace('/user-management', '');

    // サブパス別ルーティング
    switch (true) {
      case path.startsWith('/likes'):
        // いいね管理機能にリダイレクト
        const { default: likesHandler } = await import('./likes/index.ts');
        return likesHandler(req);

      case path.startsWith('/embeddings'):
        // エンベディング更新機能にリダイレクト
        const { default: embeddingsHandler } = await import('./embeddings/index.ts');
        return embeddingsHandler(req);

      case path.startsWith('/profile'):
        // プロフィール管理機能にリダイレクト
        const { default: profileHandler } = await import('./profile/index.ts');
        return profileHandler(req);

      case path.startsWith('/account'):
        // アカウント管理機能にリダイレクト
        const { default: accountHandler } = await import('./account/index.ts');
        return accountHandler(req);

      default:
        return errorResponse(
          `Unknown user management endpoint: ${path}. Available endpoints: /likes, /embeddings, /profile, /account`,
          404,
          'ENDPOINT_NOT_FOUND'
        );
    }

  } catch (error) {
    console.error('User management router error:', error);
    return errorResponse('Internal router error', 500, 'ROUTER_ERROR');
  }
});