import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface UserFeatures {
  genre_distribution: Record<string, number>;
  maker_distribution: Record<string, number>;
  performer_distribution: Record<string, number>;
  tag_distribution: Record<string, number>;
  price_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  activity_stats: {
    total_likes: number;
    days_since_first_like: number;
    likes_per_day: number;
    recent_activity: number; // likes in last 7 days
  };
  recency_weight: number[];
}

serve(async (req) => {
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
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    );

    // ユーザー認証確認
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser();
    if (authError || !user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        {
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const userId = user.id;
    console.log(`Updating user embedding for user: ${userId}`);

    // ユーザーのいいね履歴を取得
    const { data: likes, error: likesError } = await supabaseClient
      .from('likes')
      .select(`
        created_at,
        videos!inner(
          id,
          title,
          description,
          maker,
          genre,
          price,
          video_performers!inner(
            performers(name)
          ),
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (likesError) {
      console.error('Failed to fetch user likes:', likesError);
      return new Response(
        JSON.stringify({ error: 'Failed to fetch user data' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    if (!likes || likes.length === 0) {
      // 新規ユーザーにはデフォルトエンベディングを設定
      const defaultEmbedding = Array(768).fill(0);
      const { error: insertError } = await supabaseClient
        .from('user_embeddings')
        .upsert({
          user_id: userId,
          embedding: defaultEmbedding,
          updated_at: new Date().toISOString(),
        });

      if (insertError) {
        console.error('Failed to insert default embedding:', insertError);
        return new Response(
          JSON.stringify({ error: 'Failed to create user embedding' }),
          {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      return new Response(
        JSON.stringify({ 
          success: true, 
          message: 'Default embedding created for new user',
          embedding_dim: 768 
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // 特徴量抽出
    const features = extractUserFeatures(likes);
    
    // 簡易Two-Towerモデル（Phase 1: 線形結合版）
    const userEmbedding = generateUserEmbedding(features);

    // user_embeddingsテーブルを更新
    const { error: updateError } = await supabaseClient
      .from('user_embeddings')
      .upsert({
        user_id: userId,
        embedding: userEmbedding,
        updated_at: new Date().toISOString(),
      });

    if (updateError) {
      console.error('Failed to update user embedding:', updateError);
      return new Response(
        JSON.stringify({ error: 'Failed to update user embedding' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    console.log(`Successfully updated embedding for user: ${userId}`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: 'User embedding updated successfully',
        features_processed: {
          genres: Object.keys(features.genre_distribution).length,
          makers: Object.keys(features.maker_distribution).length,
          total_likes: features.activity_stats.total_likes,
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Unexpected error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});

function extractUserFeatures(likes: any[]): UserFeatures {
  const genreCount: Record<string, number> = {};
  const makerCount: Record<string, number> = {};
  const performerCount: Record<string, number> = {};
  const tagCount: Record<string, number> = {};
  const prices: number[] = [];
  const timestamps: string[] = [];

  // 各いいねから特徴量を抽出
  likes.forEach((like) => {
    const video = like.videos;
    
    // ジャンル分布
    if (video.genre) {
      genreCount[video.genre] = (genreCount[video.genre] || 0) + 1;
    }
    
    // メーカー分布
    if (video.maker) {
      makerCount[video.maker] = (makerCount[video.maker] || 0) + 1;
    }
    
    // 出演者分布
    video.video_performers?.forEach((vp: any) => {
      const performerName = vp.performers?.name;
      if (performerName) {
        performerCount[performerName] = (performerCount[performerName] || 0) + 1;
      }
    });
    
    // タグ分布
    video.video_tags?.forEach((vt: any) => {
      const tagName = vt.tags?.name;
      if (tagName) {
        tagCount[tagName] = (tagCount[tagName] || 0) + 1;
      }
    });
    
    // 価格
    if (video.price) {
      prices.push(video.price);
    }
    
    timestamps.push(like.created_at);
  });

  // 正規化
  const totalLikes = likes.length;
  const normalizeDistribution = (counts: Record<string, number>) => {
    const normalized: Record<string, number> = {};
    Object.entries(counts).forEach(([key, count]) => {
      normalized[key] = count / totalLikes;
    });
    return normalized;
  };

  // 価格統計
  const priceMean = prices.reduce((a, b) => a + b, 0) / prices.length || 0;
  const priceVariance = prices.reduce((acc, price) => acc + Math.pow(price - priceMean, 2), 0) / prices.length || 0;
  const priceStd = Math.sqrt(priceVariance);

  // アクティビティ統計
  const now = new Date();
  const firstLike = new Date(timestamps[timestamps.length - 1]);
  const daysSinceFirst = Math.max(1, (now.getTime() - firstLike.getTime()) / (1000 * 60 * 60 * 24));
  const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const recentActivity = timestamps.filter(ts => new Date(ts) > sevenDaysAgo).length;

  // 時系列重み（新しいいいねほど重要）
  const recencyWeights = timestamps.map((_, index) => {
    return Math.exp(-0.1 * index); // 指数減衰
  });

  return {
    genre_distribution: normalizeDistribution(genreCount),
    maker_distribution: normalizeDistribution(makerCount),
    performer_distribution: normalizeDistribution(performerCount),
    tag_distribution: normalizeDistribution(tagCount),
    price_stats: {
      mean: priceMean,
      std: priceStd,
      min: Math.min(...prices) || 0,
      max: Math.max(...prices) || 0,
    },
    activity_stats: {
      total_likes: totalLikes,
      days_since_first_like: daysSinceFirst,
      likes_per_day: totalLikes / daysSinceFirst,
      recent_activity: recentActivity,
    },
    recency_weight: recencyWeights,
  };
}

function generateUserEmbedding(features: UserFeatures): number[] {
  // Phase 1: 簡易版Two-Tower（線形結合）
  // 実際のモデルでは、学習済みニューラルネットワークを使用
  
  const embedding = Array(768).fill(0);
  
  // ジャンル特徴量（次元 0-99）
  let idx = 0;
  Object.entries(features.genre_distribution).forEach(([genre, weight], i) => {
    if (idx < 100) {
      embedding[idx] = weight * (1 + Math.random() * 0.1); // ノイズ追加
      idx++;
    }
  });
  
  // メーカー特徴量（次元 100-199）
  idx = 100;
  Object.entries(features.maker_distribution).forEach(([maker, weight], i) => {
    if (idx < 200) {
      embedding[idx] = weight * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  // タグ特徴量（次元 200-399）
  idx = 200;
  Object.entries(features.tag_distribution).forEach(([tag, weight], i) => {
    if (idx < 400) {
      embedding[idx] = weight * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  // 出演者特徴量（次元 400-599）
  idx = 400;
  Object.entries(features.performer_distribution).forEach(([performer, weight], i) => {
    if (idx < 600) {
      embedding[idx] = weight * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  // 価格・アクティビティ特徴量（次元 600-767）
  embedding[600] = features.price_stats.mean / 10000; // 正規化
  embedding[601] = features.price_stats.std / 10000;
  embedding[602] = Math.tanh(features.activity_stats.likes_per_day); // tanh正規化
  embedding[603] = Math.tanh(features.activity_stats.recent_activity / 10);
  
  // 時系列重み（次元 604-767）
  const maxWeights = Math.min(features.recency_weight.length, 164);
  for (let i = 0; i < maxWeights; i++) {
    embedding[604 + i] = features.recency_weight[i];
  }
  
  // L2正規化
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  if (norm > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
  }
  
  return embedding;
}