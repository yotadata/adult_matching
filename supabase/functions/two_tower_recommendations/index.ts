import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

// Inline type definition to avoid import issues
interface VideoWithTags {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url?: string;
  maker: string;
  price: number;
  sample_video_url?: string;
  image_urls: string[];
  duration_seconds?: number;
  genre: string;
  all_tags: string[];
  created_at?: string;
  updated_at?: string;
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface TwoTowerRecommendedVideo extends VideoWithTags {
  performers: string[];
  tags: string[];
  similarity_score: number;
  recommendation_reason?: string;
}

interface UserFeatures {
  user_id: string;
  total_likes: number;
  avg_price: number;
  likes_per_day: number;
  tags_preferences: number[];
}

interface VideoFeatures {
  video_id: string;
  text_features: number[];
  tags_encoded: number[];
  maker_encoded: number;
  price_normalized: number;
  duration_normalized: number;
}

/**
 * Calculate user features from interaction history
 */
async function calculateUserFeatures(
  supabaseClient: any,
  userId: string,
  tagsEncoder: Map<string, number>
): Promise<UserFeatures> {
  try {
    // ユーザーのいいね履歴を取得
    const { data: userStats } = await supabaseClient
      .from('likes')
      .select(`
        created_at,
        videos!inner(
          id,
          price,
          video_tags!inner(
            tags(name)
          )
        )
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (!userStats || userStats.length === 0) {
      return {
        user_id: userId,
        total_likes: 0,
        avg_price: 0,
        likes_per_day: 0,
        tags_preferences: Array(tagsEncoder.size).fill(0)
      };
    }

    const totalLikes = userStats.length;
    const avgPrice = userStats.reduce((sum, stat) => sum + (stat.videos?.price || 0), 0) / totalLikes;

    // Calculate tags preferences
    const tagsPreferences = Array(tagsEncoder.size).fill(0);
    const tagsCounts = new Map<string, number>();

    userStats.forEach(stat => {
      const tags = stat.videos?.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));
      if (genreTag) {
        tagsCounts.set(genreTag, (tagsCounts.get(genreTag) || 0) + 1);
      }
    });

    tagsCounts.forEach((count, tag) => {
      const tagId = tagsEncoder.get(tag);
      if (tagId !== undefined) {
        tagsPreferences[tagId] = count / totalLikes;
      }
    });

    return {
      user_id: userId,
      total_likes: totalLikes,
      avg_price: avgPrice,
      likes_per_day: totalLikes / 30,
      tags_preferences: tagsPreferences
    };
  } catch (error) {
    console.error('Error calculating user features:', error);
    return {
      user_id: userId,
      total_likes: 0,
      avg_price: 0,
      likes_per_day: 0,
      tags_preferences: Array(tagsEncoder.size).fill(0)
    };
  }
}

/**
 * Simple dot product similarity calculation
 */
function calculateDotProduct(userEmb: number[], videoEmb: number[]): number {
  if (userEmb.length !== videoEmb.length) {
    throw new Error('Embedding dimensions must match');
  }
  
  return userEmb.reduce((sum, val, idx) => sum + val * videoEmb[idx], 0);
}

/**
 * Lightweight Two-Tower inference without external models
 * This is a simplified version that demonstrates the architecture
 */
function simpleTwoTowerInference(
  userFeatures: UserFeatures,
  videoFeatures: VideoFeatures[]
): { videoId: string, score: number }[] {
  
  // Simplified User Tower (linear transformation)
  const userNumerical = [
    Math.log(userFeatures.total_likes + 1) / 10,  // Log-normalized
    userFeatures.avg_price / 10000,  // Price normalized
    Math.min(userFeatures.likes_per_day, 10) / 10  // Activity capped
  ];
  
  const userEmbedding = [
    ...userNumerical,
    ...userFeatures.tags_preferences.slice(0, 61)  // First 61 tags to make 64-dim
  ].slice(0, 64);
  
  // Pad with zeros if needed
  while (userEmbedding.length < 64) {
    userEmbedding.push(0);
  }

  // Calculate similarities
  const scores = videoFeatures.map(video => {
    // Simplified Item Tower  
    const videoNumerical = [
      video.price_normalized,
      video.duration_normalized,
      ...video.tags_encoded.slice(0, 2)  // First 2 tags encoded
    ];
    
    const videoEmbedding = [
      ...videoNumerical,
      ...video.text_features.slice(0, 60)  // Text features
    ].slice(0, 64);
    
    // Pad with zeros if needed
    while (videoEmbedding.length < 64) {
      videoEmbedding.push(0);
    }
    
    const score = calculateDotProduct(userEmbedding, videoEmbedding);
    
    return {
      videoId: video.video_id,
      score: Math.max(0, Math.min(1, (score + 1) / 2))  // Normalize to [0,1]
    };
  });

  return scores.sort((a, b) => b.score - a.score);
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const { user_id, limit = 20, exclude_liked = true } = await req.json();

    if (!user_id) {
      return new Response(
        JSON.stringify({ error: 'user_id is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    console.log(`Simple Two-Tower recommendations for user: ${user_id}`);

    // Get tags/maker encoders
    const { data: videos } = await supabaseClient
      .from('videos')
      .select(`
        video_tags!inner(
          tags(name)
        )
      `)
      .limit(100);

    const tagsEncoder = new Map<string, number>();
    const allTags = videos?.flatMap(v => v.video_tags?.map((vt: any) => vt.tags?.name)).filter(Boolean) || [];
    const uniqueTags = [...new Set(allTags)];
    uniqueTags.forEach((tag, idx) => tagsEncoder.set(tag, idx));

    // Get user features using tags-based queries
    const userFeatures = await calculateUserFeatures(supabaseClient, user_id, tagsEncoder);

    // Get candidate videos
    let candidateQuery = supabaseClient
      .from('videos')
      .select(`
        id,
        title,
        description,
        thumbnail_url,
        preview_video_url,
        maker,
        price,
        sample_video_url,
        image_urls,
        duration_seconds,
        created_at,
        updated_at,
        video_performers!inner(performers(name)),
        video_tags!inner(tags(name))
      `)
      .not('thumbnail_url', 'is', null)
      .limit(limit * 3);

    if (exclude_liked) {
      const { data: likedVideos } = await supabaseClient
        .from('likes')
        .select('video_id')
        .eq('user_id', user_id);
      
      if (likedVideos && likedVideos.length > 0) {
        const likedIds = likedVideos.map(l => l.video_id);
        candidateQuery = candidateQuery.not('id', 'in', `(${likedIds.join(',')})`)
      }
    }

    const { data: candidateVideos, error: videosError } = await candidateQuery;
    if (videosError) {
      throw new Error(`Failed to fetch videos: ${videosError.message}`);
    }

    if (!candidateVideos || candidateVideos.length === 0) {
      return new Response(
        JSON.stringify({ 
          recommendations: [],
          message: 'No candidate videos found'
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // Extract features for Two-Tower model
    const videoFeatures: VideoFeatures[] = candidateVideos.map(video => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

      return {
        video_id: video.id,
        text_features: new Array(60).fill(0).map(() => Math.random() - 0.5), // Simple random features
        tags_encoded: [tagsEncoder.get(genreTag) || 0, 0], // Tags-based encoding
        maker_encoded: Math.abs(video.maker.charCodeAt(0) % 100),
        price_normalized: video.price / 10000,
        duration_normalized: (video.duration_seconds || 3600) / 7200
      };
    });

    // Run simplified Two-Tower inference
    const scores = simpleTwoTowerInference(userFeatures, videoFeatures);

    // Build recommendations with tags-based data
    const recommendations: TwoTowerRecommendedVideo[] = scores
      .slice(0, limit)
      .map(({ videoId, score }) => {
        const video = candidateVideos.find(v => v.id === videoId)!;
        const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
        const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';

        return {
          id: video.id,
          title: video.title,
          description: video.description,
          thumbnail_url: video.thumbnail_url,
          preview_video_url: video.preview_video_url,
          maker: video.maker,
          genre: genreTag, // tagsから動的に生成されたgenre
          price: video.price,
          sample_video_url: video.sample_video_url,
          image_urls: video.image_urls || [],
          performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
          tags: tags,
          all_tags: tags, // 全タグをall_tagsとして使用
          similarity_score: score,
          recommendation_reason: `Two-Tower模型推薦 (${(score * 100).toFixed(1)}%)`,
          created_at: video.created_at,
          updated_at: video.updated_at
        };
      });

    console.log(`Generated ${recommendations.length} Two-Tower recommendations`);

    return new Response(
      JSON.stringify({
        recommendations,
        total_count: recommendations.length,
        algorithm: 'simple_two_tower_64d',
        user_features: {
          total_likes: userFeatures.total_likes,
          avg_price: userFeatures.avg_price,
          top_tags: userFeatures.tags_preferences
            .map((score, idx) => ({ tag: uniqueTags[idx], score }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 5)
            .map(t => t.tag)
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Simple Two-Tower recommendation error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error', 
        message: error.message,
        algorithm: 'simple_two_tower_64d'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});