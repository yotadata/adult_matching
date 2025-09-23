import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

/**
 * 後方互換性プロキシ関数
 * 旧 /recommendations エンドポイントへのリクエストを
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
    console.log('Legacy /recommendations endpoint called - redirecting to enhanced_two_tower');
    
    // リクエストボディを取得
    const requestBody = await req.json();
    
    // 後方互換性: 旧パラメータを新パラメータに変換
    const enhancedRequest = {
      ...requestBody,
      algorithm: 'basic', // 旧recommendations互換アルゴリズム使用
      include_reasons: requestBody.include_reasons ?? requestBody.include_explanations ?? true,
      limit: requestBody.limit ?? requestBody.max_results ?? 20
    };

    // 統合推薦関数のURLを構築
    const url = new URL(req.url);
    url.pathname = url.pathname.replace('/recommendations', '/recommendations/enhanced_two_tower');

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
        _compatibility_mode: 'legacy_recommendations',
        _redirected_to: 'enhanced_two_tower'
      }),
      {
        status: response.status,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Error in recommendations proxy:', error);
    
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error in compatibility layer',
        details: error.message 
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});