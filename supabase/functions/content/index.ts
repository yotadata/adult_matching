import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { 
  optionsResponse,
  errorResponse
} from "../_shared/responses.ts";

/**
 * コンテンツ関数グループのメインルーター
 * 
 * パス構造:
 * - /content/feed → 探索フィード (多様性重視)
 * - /content/recommendations → パーソナライズド推薦
 * - /content/videos-feed → パーソナライズド動画フィード
 * - /content/search → コンテンツ検索
 */

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  try {
    const url = new URL(req.url);
    // パス正規化：複数の形式に対応
    let path = url.pathname
      .replace(/^\/functions\/v1\/content/, '')
      .replace(/^\/content/, '');
    
    console.log(`=== Content Router Debug ===`);
    console.log(`Original URL: ${req.url}`);
    console.log(`Pathname: ${url.pathname}`);
    console.log(`Processed Path: ${path}`);
    console.log(`Method: ${req.method}`);

    // サブパス別ルーティング
    switch (true) {
      case path.startsWith('/feed'):
        console.log('Routing to feed handler...');
        // 探索フィード機能を直接実行
        return await (await import('./feed/index.ts')).default(req);

      case path.startsWith('/recommendations'):
        console.log('Routing to recommendations handler...');
        // 推薦機能を直接実行  
        return await (await import('./recommendations/index.ts')).default(req);

      case path.startsWith('/videos-feed'):
        console.log('Routing to videos-feed handler...');
        // パーソナライズド動画フィード機能を直接実行
        return await (await import('./videos-feed/index.ts')).default(req);

      case path.startsWith('/search'):
        console.log('Routing to search handler...');
        // 検索機能を直接実行
        return await (await import('./search/index.ts')).default(req);

      default:
        console.log(`No matching route found for path: ${path}`);
        return errorResponse(
          `Unknown content endpoint: ${path}. Available endpoints: /feed, /recommendations, /videos-feed, /search`,
          404,
          'ENDPOINT_NOT_FOUND'
        );
    }

  } catch (error) {
    console.error('Content router error:', error);
    console.error('Error stack:', error.stack);
    return errorResponse(`Internal router error: ${error.message}`, 500, 'ROUTER_ERROR');
  }
});