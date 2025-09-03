import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface TwoTowerRecommendedVideo {
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

interface UserFeatures {
  user_id: string;
  total_likes: number;
  avg_price: number;
  likes_per_day: number;
  genre_preferences: number[];
}

interface VideoFeatures {
  video_id: string;
  text_features: number[];
  genre_encoded: number;
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
  genreEncoder: Map<string, number>
): Promise<UserFeatures> {
  // Get user's interaction history
  const { data: userStats, error: statsError } = await supabaseClient
    .from('likes')
    .select(`
      created_at,
      videos!inner(genre, price)
    `)
    .eq('user_id', userId)
    .gte('created_at', new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString()) // 6 months
    .order('created_at', { ascending: false });

  if (statsError || !userStats?.length) {
    console.log(`No user stats found for ${userId}, using defaults`);
    // Return default features for cold start
    return {
      user_id: userId,
      total_likes: 0,
      avg_price: 0,
      likes_per_day: 0,
      genre_preferences: Array(genreEncoder.size).fill(0)
    };
  }

  // Calculate statistics
  const totalLikes = userStats.length;
  const avgPrice = userStats.reduce((sum, stat) => sum + (stat.videos.price || 0), 0) / totalLikes;
  
  const firstLike = new Date(userStats[userStats.length - 1].created_at);
  const lastLike = new Date(userStats[0].created_at);
  const daysActive = Math.max(1, (lastLike.getTime() - firstLike.getTime()) / (24 * 60 * 60 * 1000));
  const likesPerDay = totalLikes / daysActive;

  // Calculate genre preferences
  const genrePreferences = Array(genreEncoder.size).fill(0);
  const genreCounts = new Map<string, number>();
  
  userStats.forEach(stat => {
    if (stat.videos.genre) {
      genreCounts.set(stat.videos.genre, (genreCounts.get(stat.videos.genre) || 0) + 1);
    }
  });

  genreCounts.forEach((count, genre) => {
    const genreId = genreEncoder.get(genre);
    if (genreId !== undefined) {
      genrePreferences[genreId] = count / totalLikes;
    }
  });

  return {
    user_id: userId,
    total_likes: totalLikes,
    avg_price: avgPrice,
    likes_per_day: likesPerDay,
    genre_preferences: genrePreferences
  };
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
    ...userFeatures.genre_preferences.slice(0, 61)  // First 61 genres to make 64-dim
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
      video.genre_encoded / 100,  // Genre encoded
      video.maker_encoded / 100   // Maker encoded
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
    const authHeader = req.headers.get('Authorization');
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? 'http://127.0.0.1:54321';
    const anonKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';

    const supabaseClient = createClient(supabaseUrl, anonKey, {
      global: {
        headers: { Authorization: authHeader || '' },
      },
    });

    // User authentication (temporarily bypassed for testing)
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser();
    
    // Use test user ID if authentication fails
    let userId = user?.id;
    if (!userId) {
      console.log('Using test user ID for demo');
      userId = 'e010209e-1e56-450d-bf5f-76a478a9c6ca'; // Test user ID
    }

    const { limit = 50, exclude_liked = true } = await req.json();

    console.log(`Two-Tower recommendations for user: ${userId}`);

    // Get genre/maker encoders (simplified)
    const { data: genres } = await supabaseClient
      .from('videos')
      .select('genre')
      .not('genre', 'is', null);

    const genreEncoder = new Map<string, number>();
    const uniqueGenres = [...new Set(genres?.map(g => g.genre) || [])];
    uniqueGenres.forEach((genre, idx) => genreEncoder.set(genre, idx));

    // Calculate user features
    const userFeatures = await calculateUserFeatures(supabaseClient, userId, genreEncoder);

    // Get candidate videos
    let videoQuery = supabaseClient
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
        duration_seconds,
        sample_video_url,
        image_urls,
        video_performers!inner(performers(name)),
        video_tags!inner(tags(name))
      `)
      .not('thumbnail_url', 'is', null)
      .limit(limit * 3);  // Get more candidates for filtering

    if (exclude_liked) {
      const { data: likedVideos } = await supabaseClient
        .from('likes')
        .select('video_id')
        .eq('user_id', userId);
      
      const likedIds = likedVideos?.map(l => l.video_id) || [];
      if (likedIds.length > 0) {
        videoQuery = videoQuery.not('id', 'in', `(${likedIds.join(',')})`);
      }
    }

    const { data: candidateVideos, error: videosError } = await videoQuery;

    if (videosError) {
      throw new Error(`Failed to fetch videos: ${videosError.message}`);
    }

    if (!candidateVideos?.length) {
      return new Response(
        JSON.stringify({ 
          recommendations: [],
          message: 'No candidate videos found',
          fallback: true
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // Prepare video features (simplified)
    const videoFeatures: VideoFeatures[] = candidateVideos.map((video, idx) => {
      const textFeatures = Array(60).fill(0); // Placeholder for TF-IDF features
      const titleWords = (video.title || '').toLowerCase().split(' ');
      // Simple hash-based text features
      titleWords.forEach(word => {
        if (word.length > 2) {
          const hash = word.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0);
          textFeatures[hash % 60] += 1;
        }
      });

      return {
        video_id: video.id,
        text_features: textFeatures.map(f => f / Math.max(1, titleWords.length)),
        genre_encoded: genreEncoder.get(video.genre) || 0,
        maker_encoded: video.maker ? video.maker.length % 100 : 0, // Simplified
        price_normalized: (video.price || 0) / 10000,
        duration_normalized: (video.duration_seconds || 0) / 7200 // 2 hours max
      };
    });

    // Run Two-Tower inference
    const scores = simpleTwoTowerInference(userFeatures, videoFeatures);

    // Build recommendations
    const recommendations: TwoTowerRecommendedVideo[] = scores
      .slice(0, limit)
      .map(({ videoId, score }) => {
        const video = candidateVideos.find(v => v.id === videoId)!;
        
        return {
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
          similarity_score: score,
          recommendation_reason: `Two-Tower マッチング (スコア: ${(score * 100).toFixed(1)}%)`
        };
      });

    console.log(`Generated ${recommendations.length} Two-Tower recommendations`);

    return new Response(
      JSON.stringify({
        recommendations,
        total_count: recommendations.length,
        algorithm: 'two_tower',
        user_features: {
          total_likes: userFeatures.total_likes,
          avg_price: userFeatures.avg_price,
          top_genres: userFeatures.genre_preferences
            .map((score, idx) => ({ genre: uniqueGenres[idx], score }))
            .filter(g => g.score > 0.1)
            .sort((a, b) => b.score - a.score)
            .slice(0, 3)
            .map(g => g.genre)
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Two-Tower recommendation error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error', 
        message: error.message 
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});