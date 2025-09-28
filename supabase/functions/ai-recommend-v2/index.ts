import "https://deno.land/x/xhr@0.3.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

// Two-Tower モデルの重みとロジックをインポート
import './two_tower_model.js';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// グローバルスコープからTwo-Towerモデル関数を取得
declare global {
  var TwoTowerModel: {
    predictUserEmbedding: (userFeatures: number[]) => number[];
    predictItemEmbedding: (itemFeatures: number[]) => number[];
    cosineSimilarity: (vec1: number[], vec2: number[]) => number;
    preprocessUserFeatures: (rawFeatures: any) => number[];
    preprocessItemFeatures: (rawFeatures: any) => number[];
  };
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  if (req.method !== 'GET' && req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405, headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get('Authorization') ?? '';
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      { global: { headers: { Authorization: authHeader } } }
    );

    // ユーザー情報取得
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      // 未認証ユーザー向けフォールバック（トレンディング動画のみ）
      return await handleUnauthenticatedRequest(supabase);
    }

    // 認証済みユーザー向け処理
    return await handleAuthenticatedRequest(supabase, user);

  } catch (err: any) {
    console.error('[ai-recommend-v2] unexpected error:', err?.message || err);
    return new Response(JSON.stringify({ error: err?.message || 'unknown error' }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});

async function handleUnauthenticatedRequest(supabase: any) {
  // トレンディング動画のみ返す
  const { data: trending, error } = await supabase
    .from('videos')
    .select('id, title, description, external_id, thumbnail_url, created_at')
    .order('created_at', { ascending: false })
    .limit(10);

  if (error) {
    throw new Error(`Trending videos fetch error: ${error.message}`);
  }

  return new Response(JSON.stringify({
    personalized: [],
    trending: trending || []
  }), {
    status: 200,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

async function handleAuthenticatedRequest(supabase: any, user: any) {
  console.log('[ai-recommend-v2] Processing authenticated request for user:', user.id);

  // ユーザープロファイル取得
  const { data: profile } = await supabase
    .from('profiles')
    .select('display_name, created_at')
    .eq('user_id', user.id)
    .single();

  // ユーザーの決定履歴を取得
  const { count: totalDecisions } = await supabase
    .from('user_video_decisions')
    .select('*', { count: 'exact', head: true })
    .eq('user_id', user.id);

  const { count: likeCount } = await supabase
    .from('user_video_decisions')
    .select('*', { count: 'exact', head: true })
    .eq('user_id', user.id)
    .eq('decision_type', 'like');

  // ユーザーの追加統計情報を取得（価格帯好み等）
  const { data: likedVideos } = await supabase
    .from('user_video_decisions')
    .select(`
      video_id,
      videos!inner(price, tags)
    `)
    .eq('user_id', user.id)
    .eq('decision_type', 'like');

  // 価格帯好み計算
  const likedPrices = likedVideos?.map(v => v.videos?.price).filter(p => p != null) || [];
  const avgLikedPrice = likedPrices.length > 0 ?
    likedPrices.reduce((sum, price) => sum + price, 0) / likedPrices.length : 0;
  const pricePreferenceHigh = likedPrices.length > 0 ?
    likedPrices.filter(price => price > 1000).length / likedPrices.length : 0;

  // ユーザー特徴量を作成（13次元）
  const userFeatures = {
    display_name: profile?.display_name || '',
    account_age_days: profile?.created_at ?
      Math.floor((Date.now() - new Date(profile.created_at).getTime()) / (1000 * 60 * 60 * 24)) : 0,
    total_decisions: totalDecisions || 0,
    like_ratio: totalDecisions ? (likeCount || 0) / totalDecisions : 0.5,
    avg_liked_price: avgLikedPrice,
    price_preference_high: pricePreferenceHigh,
    like_count: likeCount || 0,
    nope_count: (totalDecisions || 0) - (likeCount || 0)
  };

  console.log('[ai-recommend-v2] User features:', userFeatures);

  // ユーザー埋め込みベクトルを計算
  const userEmbedding = globalThis.TwoTowerModel.predictUserEmbedding(
    globalThis.TwoTowerModel.preprocessUserFeatures(userFeatures)
  );

  console.log('[ai-recommend-v2] User embedding computed, dimension:', userEmbedding.length);

  // 候補動画を取得（視聴履歴を除外）
  const { data: watchedVideos } = await supabase
    .from('user_video_decisions')
    .select('video_id')
    .eq('user_id', user.id);

  const watchedVideoIds = watchedVideos?.map((w: any) => w.video_id) || [];

  let candidateQuery = supabase
    .from('videos')
    .select('id, title, description, external_id, thumbnail_url, created_at, price, product_released_at, maker, director, series, tags, sample_video_url, preview_video_url, image_urls, duration_seconds')
    .order('created_at', { ascending: false })
    .limit(200);

  if (watchedVideoIds.length > 0) {
    candidateQuery = candidateQuery.not('id', 'in', `(${watchedVideoIds.join(',')})`);
  }

  const { data: candidateVideos, error: videosError } = await candidateQuery;

  if (videosError) {
    throw new Error(`Videos fetch error: ${videosError.message}`);
  }

  console.log('[ai-recommend-v2] Candidate videos:', candidateVideos?.length || 0);

  // Two-Tower推論による推薦スコア計算
  const scoredVideos = candidateVideos?.map((video: any) => {
    // 動画の人気統計情報を追加（簡易版、実際は別途計算）
    const videoWithStats = {
      ...video,
      total_decisions: 10, // 暫定値
      like_ratio: 0.8     // 暫定値
    };

    const itemEmbedding = globalThis.TwoTowerModel.predictItemEmbedding(
      globalThis.TwoTowerModel.preprocessItemFeatures(videoWithStats)
    );
    const similarity = globalThis.TwoTowerModel.cosineSimilarity(userEmbedding, itemEmbedding);

    return {
      ...video,
      similarity_score: similarity
    };
  }) || [];

  // 類似度順にソート
  scoredVideos.sort((a, b) => b.similarity_score - a.similarity_score);

  const personalizedVideos = scoredVideos.slice(0, 10);

  console.log('[ai-recommend-v2] Top recommendation scores:',
    personalizedVideos.slice(0, 3).map(v => v.similarity_score));

  // トレンディング動画（人気順）
  const { data: trendingVideos } = await supabase
    .from('videos')
    .select('id, title, description, external_id, thumbnail_url, created_at')
    .order('created_at', { ascending: false })
    .limit(10);

  return new Response(JSON.stringify({
    personalized: personalizedVideos,
    trending: trendingVideos || [],
    debug: {
      user_features: userFeatures,
      embedding_dim: userEmbedding.length,
      candidate_count: candidateVideos?.length || 0,
      top_scores: personalizedVideos.slice(0, 5).map(v => ({
        id: v.id,
        title: v.title?.substring(0, 50),
        score: v.similarity_score
      }))
    }
  }), {
    status: 200,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}