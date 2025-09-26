import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

/**
 * 後方互換性プロキシ関数
 * 旧 /two_tower_recommendations エンドポイントへのリクエストを
 * 新 /recommendations/enhanced_two_tower にリダイレクト
 */

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  if (req.method !== 'POST') {
    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      {
        status: 405,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }

  try {
    console.log('Legacy /two_tower_recommendations endpoint called - redirecting to enhanced_two_tower');
    
    // リクエストボディを取得
    const requestBody = await req.json();
    
    // 後方互換性: 旧two_tower_recommendationsパラメータを新パラメータに変換
    const enhancedRequest = {
      ...requestBody,
      algorithm: 'two_tower', // 旧two_tower_recommendations互換アルゴリズム使用
      include_reasons: requestBody.include_reasons ?? true,
      limit: requestBody.limit ?? 20,
      // 64次元→768次元への移行通知
      _migration_notice: '64次元から768次元Two-Towerアルゴリズムに自動アップグレードされました'
    };

    // 統合推薦関数のURLを構築
    const url = new URL(req.url);
    url.pathname = '/functions/v1/recommendations/enhanced_two_tower';

    // ヘッダーをコピー
    const headers = new Headers(req.headers);
    headers.set('Content-Type', 'application/json');

    // enhanced_two_tower エンドポイントに転送
    const response = await fetch(url.toString(), {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(enhancedRequest)
    });

    // レスポンスをそのまま返す
    const responseData = await response.json();
    
    return new Response(
      JSON.stringify({
        ...responseData,
        // デバッグ情報追加
        _compatibility_mode: 'legacy_two_tower',
        _redirected_to: 'enhanced_two_tower',
        _algorithm_upgrade: '64dim -> 768dim'
      }),
      {
        status: response.status,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Error in two_tower_recommendations proxy:', error);
    
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error in compatibility layer',
        details: error.message,
        legacy_endpoint: 'two_tower_recommendations'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});