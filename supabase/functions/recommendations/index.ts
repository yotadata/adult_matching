import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface RecommendedVideo {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
  similarity_score: number;
  recommendation_reason?: string;
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

    const { limit = 20, exclude_liked = true } = await req.json();
    const userId = user.id;

    console.log(`Generating recommendations for user: ${userId}`);

    // ユーザーエンベディングを取得
    const { data: userEmbedding, error: userEmbeddingError } = await supabaseClient
      .from('user_embeddings')
      .select('embedding, updated_at')
      .eq('user_id', userId)
      .single();

    if (userEmbeddingError || !userEmbedding) {
      console.log('User embedding not found, triggering update...');
      
      // ユーザーエンベディングが存在しない場合、まず更新を実行
      const updateResponse = await supabaseClient.functions.invoke('update_user_embedding');
      if (updateResponse.error) {
        console.error('Failed to update user embedding:', updateResponse.error);
      }
      
      // フォールバック：多様性重視の推奨を返す
      return await generateDiverseRecommendations(supabaseClient, userId, limit, exclude_liked);
    }

    // いいね済み動画IDを取得（除外用）
    let excludeVideoIds: string[] = [];
    if (exclude_liked) {
      const { data: likedVideos } = await supabaseClient
        .from('likes')
        .select('video_id')
        .eq('user_id', userId);
      
      excludeVideoIds = likedVideos?.map(like => like.video_id) || [];
    }

    // 全動画エンベディングを取得
    const { data: videoEmbeddings, error: videoEmbeddingsError } = await supabaseClient
      .from('video_embeddings')
      .select(`
        video_id,
        embedding,
        videos!inner(
          id,
          title,
          description,
          thumbnail_url,
          preview_video_url,
          maker,
          genre,
          price,
          sample_video_url,
          image_urls,
          video_performers!inner(
            performers(name)
          ),
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .not('video_id', 'in', `(${excludeVideoIds.join(',')})`);

    if (videoEmbeddingsError || !videoEmbeddings) {
      console.error('Failed to fetch video embeddings:', videoEmbeddingsError);
      return new Response(
        JSON.stringify({ error: 'Failed to fetch video data' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // コサイン類似度計算
    const recommendations: RecommendedVideo[] = [];
    const userVector = userEmbedding.embedding;

    for (const videoData of videoEmbeddings) {
      const videoVector = videoData.embedding;
      const similarity = calculateCosineSimilarity(userVector, videoVector);
      
      const video = videoData.videos;
      const recommendedVideo: RecommendedVideo = {
        id: video.id,
        title: video.title,
        description: video.description,
        thumbnail_url: video.thumbnail_url,
        preview_video_url: video.preview_video_url,
        maker: video.maker,
        genre: video.genre,
        price: video.price,
        sample_video_url: video.sample_video_url,
        image_urls: video.image_urls || [],
        performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
        tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
        similarity_score: similarity,
        recommendation_reason: generateRecommendationReason(similarity, video),
      };
      
      recommendations.push(recommendedVideo);
    }

    // 類似度でソート
    recommendations.sort((a, b) => b.similarity_score - a.similarity_score);

    // 多様性を考慮した選択（トップの中からジャンル・メーカーのバランスを調整）
    const diverseRecommendations = selectDiverseRecommendations(
      recommendations.slice(0, Math.min(100, recommendations.length)), 
      limit
    );

    console.log(`Generated ${diverseRecommendations.length} recommendations for user: ${userId}`);

    return new Response(
      JSON.stringify({
        recommendations: diverseRecommendations,
        total_candidates: videoEmbeddings.length,
        user_embedding_updated: userEmbedding.updated_at,
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

function calculateCosineSimilarity(vectorA: number[], vectorB: number[]): number {
  if (vectorA.length !== vectorB.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    normA += vectorA[i] * vectorA[i];
    normB += vectorB[i] * vectorB[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}

function generateRecommendationReason(similarity: number, video: any): string {
  if (similarity > 0.8) {
    return `あなたの嗜好と非常に合致しています（類似度: ${Math.round(similarity * 100)}%）`;
  } else if (similarity > 0.6) {
    return `${video.genre}ジャンルでおすすめです`;
  } else if (similarity > 0.4) {
    return `${video.maker}の作品です`;
  } else {
    return `新しいジャンルの発見かも？`;
  }
}

function selectDiverseRecommendations(
  recommendations: RecommendedVideo[], 
  limit: number
): RecommendedVideo[] {
  const selected: RecommendedVideo[] = [];
  const usedGenres = new Set<string>();
  const usedMakers = new Set<string>();
  
  // 第1段階: 高類似度 + 多様性
  for (const rec of recommendations) {
    if (selected.length >= limit) break;
    
    const genreUsed = usedGenres.has(rec.genre);
    const makerUsed = usedMakers.has(rec.maker);
    
    // 上位は多様性を重視
    if (selected.length < limit * 0.7) {
      if (!genreUsed || !makerUsed) {
        selected.push(rec);
        usedGenres.add(rec.genre);
        usedMakers.add(rec.maker);
        continue;
      }
    }
    
    // 残りは類似度重視
    if (rec.similarity_score > 0.3) {
      selected.push(rec);
    }
  }
  
  // 不足分は上位から補完
  while (selected.length < limit && selected.length < recommendations.length) {
    const candidate = recommendations[selected.length];
    if (!selected.includes(candidate)) {
      selected.push(candidate);
    }
  }
  
  return selected.slice(0, limit);
}

async function generateDiverseRecommendations(
  supabaseClient: any, 
  userId: string, 
  limit: number, 
  excludeLiked: boolean
): Promise<Response> {
  console.log('Generating diverse recommendations (fallback)');
  
  let excludeVideoIds: string[] = [];
  if (excludeLiked) {
    const { data: likedVideos } = await supabaseClient
      .from('likes')
      .select('video_id')
      .eq('user_id', userId);
    
    excludeVideoIds = likedVideos?.map((like: any) => like.video_id) || [];
  }
  
  const { data: videos, error } = await supabaseClient
    .from('videos')
    .select(`
      id,
      title,
      description,
      thumbnail_url,
      preview_video_url,
      maker,
      genre,
      price,
      sample_video_url,
      image_urls,
      video_performers!inner(
        performers(name)
      ),
      video_tags!inner(
        tags(name)
      )
    `)
    .not('id', 'in', `(${excludeVideoIds.join(',')})`)
    .order('published_at', { ascending: false })
    .limit(limit * 3);

  if (error || !videos) {
    return new Response(
      JSON.stringify({ error: 'Failed to fetch fallback recommendations' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }

  const recommendations: RecommendedVideo[] = videos.map((video: any) => ({
    id: video.id,
    title: video.title,
    description: video.description,
    thumbnail_url: video.thumbnail_url,
    preview_video_url: video.preview_video_url,
    maker: video.maker,
    genre: video.genre,
    price: video.price,
    sample_video_url: video.sample_video_url,
    image_urls: video.image_urls || [],
    performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
    tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
    similarity_score: Math.random() * 0.5 + 0.3, // ランダムスコア
    recommendation_reason: '多様性重視の推奨です',
  }));

  const diverseRecs = selectDiverseRecommendations(recommendations, limit);

  return new Response(
    JSON.stringify({
      recommendations: diverseRecs,
      total_candidates: videos.length,
      fallback: true,
      message: 'ユーザーエンベディングを更新中です。次回はより精度の高い推奨を提供します。',
    }),
    {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    }
  );
}