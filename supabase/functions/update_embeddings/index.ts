import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";
import {
  VIDEOS_WITH_TAGS_FULL_SELECT,
  getUserGenrePreferencesQuery,
  getVideoGenresBatch
} from '../_shared/query-patterns.ts';
import { VideoWithTags } from '../_shared/types.ts';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface BatchUpdateRequest {
  update_type: 'videos' | 'users' | 'both';
  batch_size?: number;
  force_update?: boolean;
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
    // Service role key required for batch operations
    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false
        }
      }
    );

    const { 
      update_type = 'both', 
      batch_size = 100, 
      force_update = false 
    }: BatchUpdateRequest = await req.json();

    console.log(`Starting batch embedding update: ${update_type}`);

    const results = {
      videos_updated: 0,
      users_updated: 0,
      errors: [] as string[],
      start_time: new Date().toISOString(),
    };

    // Video embeddings update
    if (update_type === 'videos' || update_type === 'both') {
      try {
        const videosUpdated = await updateVideoEmbeddings(supabaseAdmin, batch_size, force_update);
        results.videos_updated = videosUpdated;
        console.log(`Updated ${videosUpdated} video embeddings`);
      } catch (error) {
        console.error('Video embedding update failed:', error);
        results.errors.push(`Video update failed: ${error.message}`);
      }
    }

    // User embeddings update
    if (update_type === 'users' || update_type === 'both') {
      try {
        const usersUpdated = await updateUserEmbeddings(supabaseAdmin, batch_size, force_update);
        results.users_updated = usersUpdated;
        console.log(`Updated ${usersUpdated} user embeddings`);
      } catch (error) {
        console.error('User embedding update failed:', error);
        results.errors.push(`User update failed: ${error.message}`);
      }
    }

    const endTime = new Date().toISOString();
    const duration = new Date(endTime).getTime() - new Date(results.start_time).getTime();

    return new Response(
      JSON.stringify({
        success: true,
        results: {
          ...results,
          end_time: endTime,
          duration_ms: duration,
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Batch update failed:', error);
    return new Response(
      JSON.stringify({ error: 'Batch update failed', details: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});

async function updateVideoEmbeddings(
  supabaseAdmin: any,
  batchSize: number,
  forceUpdate: boolean
): Promise<number> {
  // Get videos that need embedding updates using simplified query
  let query = supabaseAdmin
    .from('videos')
    .select(`
      id,
      title,
      description,
      thumbnail_url,
      maker,
      price,
      duration_seconds,
      video_performers!inner(
        performers(name)
      ),
      video_tags!inner(
        tags(name)
      )
    `);

  if (!forceUpdate) {
    // Only update videos without embeddings or old embeddings
    const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    query = query.select(`
      *,
      video_embeddings!left(updated_at)
    `).or(`video_embeddings.updated_at.is.null,video_embeddings.updated_at.lt.${oneWeekAgo}`);
  }

  const { data: videos, error } = await query.limit(batchSize);

  if (error || !videos) {
    throw new Error(`Failed to fetch videos: ${error?.message}`);
  }

  let updatedCount = 0;

  for (const video of videos) {
    try {
      const embedding = generateVideoEmbedding(video);

      const { error: upsertError } = await supabaseAdmin
        .from('video_embeddings')
        .upsert({
          video_id: video.id,
          embedding: embedding,
          updated_at: new Date().toISOString(),
        });

      if (upsertError) {
        console.error(`Failed to update embedding for video ${video.id}:`, upsertError);
        continue;
      }

      updatedCount++;
    } catch (error) {
      console.error(`Error processing video ${video.id}:`, error);
      continue;
    }
  }

  return updatedCount;
}

async function updateUserEmbeddings(
  supabaseAdmin: any, 
  batchSize: number, 
  forceUpdate: boolean
): Promise<number> {
  // Get users with recent activity that need embedding updates
  let query = supabaseAdmin
    .from('likes')
    .select('user_id, created_at')
    .order('created_at', { ascending: false });

  if (!forceUpdate) {
    const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    query = query.gte('created_at', oneWeekAgo);
  }

  const { data: recentActivity, error } = await query.limit(batchSize * 3);

  if (error) {
    throw new Error(`Failed to fetch user activity: ${error.message}`);
  }

  // Get unique user IDs
  const uniqueUserIds = [...new Set(recentActivity?.map(activity => activity.user_id) || [])];
  const userIds = uniqueUserIds.slice(0, batchSize);

  let updatedCount = 0;

  for (const userId of userIds) {
    try {
      // Use the existing update_user_embedding function logic
      await updateSingleUserEmbedding(supabaseAdmin, userId);
      updatedCount++;
    } catch (error) {
      console.error(`Failed to update embedding for user ${userId}:`, error);
      continue;
    }
  }

  return updatedCount;
}

async function updateSingleUserEmbedding(supabaseAdmin: any, userId: string) {
  // Fetch user's likes with video data using simplified query
  const { data: likes, error: likesError } = await supabaseAdmin
    .from('likes')
    .select(`
      created_at,
      videos!inner(
        id,
        title,
        description,
        maker,
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

  if (likesError || !likes || likes.length === 0) {
    // Default embedding for users without likes
    const defaultEmbedding = Array(768).fill(0);
    await supabaseAdmin
      .from('user_embeddings')
      .upsert({
        user_id: userId,
        embedding: defaultEmbedding,
        updated_at: new Date().toISOString(),
      });
    return;
  }

  // Extract features and generate embedding (same logic as update_user_embedding)
  const features = extractUserFeatures(likes);
  const embedding = generateUserEmbedding(features);

  await supabaseAdmin
    .from('user_embeddings')
    .upsert({
      user_id: userId,
      embedding: embedding,
      updated_at: new Date().toISOString(),
    });
}

function generateVideoEmbedding(video: any): number[] {
  // Phase 1: Tags-based item tower implementation
  // In production, this would use a trained neural network

  const embedding = Array(768).fill(0);

  // Text features (title + description)
  const text = `${video.title} ${video.description || ''}`.toLowerCase();
  const textHash = simpleHash(text);
  for (let i = 0; i < 100; i++) {
    embedding[i] = ((textHash >> i) & 1) * 0.1 + Math.random() * 0.05;
  }

  // Tags-based genre features (100-199) - derived from tag groups
  const allTags = video.all_tags || [];
  const genreTags = allTags.filter((tag: string) => tag && tag.toLowerCase().includes('genre'));
  if (genreTags.length > 0) {
    const primaryGenre = genreTags[0];
    const genreHash = simpleHash(primaryGenre);
    for (let i = 100; i < 200; i++) {
      embedding[i] = ((genreHash >> (i - 100)) & 1) * 0.2 + Math.random() * 0.05;
    }
  }

  // Maker features (200-299)
  if (video.maker) {
    const makerHash = simpleHash(video.maker);
    for (let i = 200; i < 300; i++) {
      embedding[i] = ((makerHash >> (i - 200)) & 1) * 0.2 + Math.random() * 0.05;
    }
  }

  // Performer features (300-499)
  const performers = video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [];
  performers.forEach((performer: string, idx: number) => {
    if (idx < 20) {
      const performerHash = simpleHash(performer);
      for (let i = 0; i < 10; i++) {
        const embedIdx = 300 + idx * 10 + i;
        if (embedIdx < 500) {
          embedding[embedIdx] = ((performerHash >> i) & 1) * 0.15 + Math.random() * 0.05;
        }
      }
    }
  });

  // All tag features (500-699) - enhanced with all_tags
  const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
  const combinedTags = [...new Set([...tags, ...allTags])].filter(Boolean);
  combinedTags.forEach((tag: string, idx: number) => {
    if (idx < 20) {
      const tagHash = simpleHash(tag);
      for (let i = 0; i < 10; i++) {
        const embedIdx = 500 + idx * 10 + i;
        if (embedIdx < 700) {
          embedding[embedIdx] = ((tagHash >> i) & 1) * 0.15 + Math.random() * 0.05;
        }
      }
    }
  });

  // Numerical features (700-767)
  embedding[700] = (video.price || 0) / 10000; // normalized price
  embedding[701] = (video.duration_seconds || 0) / 3600; // normalized duration

  // Random features for remaining dimensions
  for (let i = 702; i < 768; i++) {
    embedding[i] = Math.random() * 0.1 - 0.05;
  }

  // L2 normalization
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  if (norm > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
  }

  return embedding;
}

function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash;
}

// Re-implement user feature extraction (same as in update_user_embedding)
function extractUserFeatures(likes: any[]): any {
  const genreCount: Record<string, number> = {};
  const makerCount: Record<string, number> = {};
  const performerCount: Record<string, number> = {};
  const tagCount: Record<string, number> = {};
  const prices: number[] = [];
  const timestamps: string[] = [];

  likes.forEach((like) => {
    const video = like.videos;
    
    // Extract genre from tags
    const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
    const genreTag = tags.find((tag: string) => tag && tag.toLowerCase().includes('genre'));
    if (genreTag) {
      genreCount[genreTag] = (genreCount[genreTag] || 0) + 1;
    }
    
    if (video.maker) {
      makerCount[video.maker] = (makerCount[video.maker] || 0) + 1;
    }
    
    video.video_performers?.forEach((vp: any) => {
      const performerName = vp.performers?.name;
      if (performerName) {
        performerCount[performerName] = (performerCount[performerName] || 0) + 1;
      }
    });
    
    video.video_tags?.forEach((vt: any) => {
      const tagName = vt.tags?.name;
      if (tagName) {
        tagCount[tagName] = (tagCount[tagName] || 0) + 1;
      }
    });
    
    if (video.price) {
      prices.push(video.price);
    }
    
    timestamps.push(like.created_at);
  });

  const totalLikes = likes.length;
  const normalizeDistribution = (counts: Record<string, number>) => {
    const normalized: Record<string, number> = {};
    Object.entries(counts).forEach(([key, count]) => {
      normalized[key] = count / totalLikes;
    });
    return normalized;
  };

  const priceMean = prices.reduce((a, b) => a + b, 0) / prices.length || 0;
  const priceVariance = prices.reduce((acc, price) => acc + Math.pow(price - priceMean, 2), 0) / prices.length || 0;
  const priceStd = Math.sqrt(priceVariance);

  const now = new Date();
  const firstLike = new Date(timestamps[timestamps.length - 1]);
  const daysSinceFirst = Math.max(1, (now.getTime() - firstLike.getTime()) / (1000 * 60 * 60 * 24));
  const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const recentActivity = timestamps.filter(ts => new Date(ts) > sevenDaysAgo).length;

  const recencyWeights = timestamps.map((_, index) => {
    return Math.exp(-0.1 * index);
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

// Re-implement user embedding generation (same as in update_user_embedding)
function generateUserEmbedding(features: any): number[] {
  const embedding = Array(768).fill(0);
  
  let idx = 0;
  Object.entries(features.genre_distribution).forEach(([genre, weight], i) => {
    if (idx < 100) {
      embedding[idx] = (weight as number) * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  idx = 100;
  Object.entries(features.maker_distribution).forEach(([maker, weight], i) => {
    if (idx < 200) {
      embedding[idx] = (weight as number) * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  idx = 200;
  Object.entries(features.tag_distribution).forEach(([tag, weight], i) => {
    if (idx < 400) {
      embedding[idx] = (weight as number) * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  idx = 400;
  Object.entries(features.performer_distribution).forEach(([performer, weight], i) => {
    if (idx < 600) {
      embedding[idx] = (weight as number) * (1 + Math.random() * 0.1);
      idx++;
    }
  });
  
  embedding[600] = features.price_stats.mean / 10000;
  embedding[601] = features.price_stats.std / 10000;
  embedding[602] = Math.tanh(features.activity_stats.likes_per_day);
  embedding[603] = Math.tanh(features.activity_stats.recent_activity / 10);
  
  const maxWeights = Math.min(features.recency_weight.length, 164);
  for (let i = 0; i < maxWeights; i++) {
    embedding[604 + i] = features.recency_weight[i];
  }
  
  const norm = Math.sqrt(embedding.reduce((sum: number, val: number) => sum + val * val, 0));
  if (norm > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }
  }
  
  return embedding;
}