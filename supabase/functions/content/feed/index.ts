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

interface VideoData extends VideoWithTags {
  performers: string[];
  tags: string[];
}

interface FeedRequest {
  feed_type: 'explore' | 'personalized' | 'latest' | 'popular' | 'random';
  limit?: number;
  offset?: number;
  exclude_ids?: string[];
  user_id?: string;
}

interface FeedResponse {
  videos: VideoData[];
  total_count: number;
  feed_type: string;
  diversity_score?: number;
  pagination: {
    limit: number;
    offset: number;
    has_more: boolean;
  };
  processing_time_ms: number;
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

async function handleFeedRequest(req: Request): Promise<Response> {
  const startTime = performance.now();

  // Handle CORS preflight requests
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

    const { 
      feed_type = 'explore', 
      limit = 20, 
      offset = 0, 
      exclude_ids = [], 
      user_id 
    }: FeedRequest = await req.json();

    console.log(`=== Content Feed Request ===`);
    console.log(`Feed type: ${feed_type}, Limit: ${limit}, Offset: ${offset}, User: ${user_id || 'anonymous'}`);

    let videos: VideoData[] = [];
    let totalCount = 0;
    let diversityScore = 0;

    switch (feed_type) {
      case 'personalized':
        if (user_id) {
          // Use existing RPC function for personalized feed
          console.log('Fetching personalized feed via RPC...');
          const { data: personalizedVideos, error } = await supabaseClient.rpc('get_videos_feed', { 
            page_limit: limit,
            page_offset: offset 
          });

          if (error) {
            console.error('Error fetching personalized videos via RPC:', error.message);
            throw new Error(`Failed to fetch personalized feed: ${error.message}`);
          }

          videos = personalizedVideos || [];
          totalCount = videos.length;
          console.log(`📱 Personalized feed: ${videos.length} videos`);
        } else {
          // Fallback to explore feed for anonymous users
          console.log('No user_id provided, falling back to explore feed...');
          const exploreResult = await getExploreFeed(supabaseClient, limit, offset, exclude_ids);
          videos = exploreResult.videos;
          totalCount = exploreResult.totalCount;
          diversityScore = exploreResult.diversityScore;
        }
        break;

      case 'explore':
        console.log('Fetching explore feed with diversity strategies...');
        const exploreResult = await getExploreFeed(supabaseClient, limit, offset, exclude_ids);
        videos = exploreResult.videos;
        totalCount = exploreResult.totalCount;
        diversityScore = exploreResult.diversityScore;
        break;

      case 'latest':
        console.log('Fetching latest videos...');
        const latestResult = await getLatestFeed(supabaseClient, limit, offset, exclude_ids);
        videos = latestResult.videos;
        totalCount = latestResult.totalCount;
        break;

      case 'popular':
        console.log('Fetching popular videos...');
        const popularResult = await getPopularFeed(supabaseClient, limit, offset, exclude_ids);
        videos = popularResult.videos;
        totalCount = popularResult.totalCount;
        break;

      case 'random':
        console.log('Fetching random videos...');
        const randomResult = await getRandomFeed(supabaseClient, limit, offset, exclude_ids);
        videos = randomResult.videos;
        totalCount = randomResult.totalCount;
        break;

      default:
        throw new Error(`Invalid feed_type: ${feed_type}`);
    }

    const processingTime = performance.now() - startTime;
    console.log(`✅ Feed generation complete: ${videos.length} videos in ${processingTime.toFixed(2)}ms`);

    const response: FeedResponse = {
      videos,
      total_count: totalCount,
      feed_type,
      diversity_score: diversityScore > 0 ? diversityScore : undefined,
      pagination: {
        limit,
        offset,
        has_more: videos.length === limit, // Estimate based on returned count
      },
      processing_time_ms: processingTime,
    };

    return new Response(
      JSON.stringify(response),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Content feed error:', error);
    
    const processingTime = performance.now() - startTime;
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        processing_time_ms: processingTime
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
}

// Explore Feed with Diversity Strategies (from feed_explore)
async function getExploreFeed(supabaseClient: any, limit: number, offset: number, excludeIds: string[]) {
  console.log(`Fetching ${limit} diverse videos with offset ${offset}`);

  // 多様性を重視した動画取得戦略（tags-basedクエリ使用）
  const diversityStrategies = [
    // ジャンル分散 (40% of limit)
    () => supabaseClient
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
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `)
      .not('id', 'in', excludeIds)
      .order('created_at', { ascending: false })
      .range(offset, offset + Math.floor(limit * 0.4)),

    // 人気度順 (30% of limit)
    () => supabaseClient
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
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `)
      .not('id', 'in', excludeIds)
      .order('created_at', { ascending: false })
      .range(offset, offset + Math.floor(limit * 0.3)),

    // ランダム (30% of limit)
    () => supabaseClient
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
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `)
      .not('id', 'in', excludeIds)
      .limit(Math.ceil(limit * 0.3))
  ];

  let allVideos: any[] = [];
  
  // 各戦略から動画を取得
  for (const strategy of diversityStrategies) {
    const { data: videos, error } = await strategy();
    if (!error && videos) {
      allVideos = allVideos.concat(videos);
    }
  }

  if (allVideos.length === 0) {
    return { videos: [], totalCount: 0, diversityScore: 0 };
  }

  // 重複除去
  const uniqueVideos = allVideos.reduce((acc: any[], current: any) => {
    const isDuplicate = acc.some(video => video.id === current.id);
    if (!isDuplicate) {
      acc.push(current);
    }
    return acc;
  }, []);

  // データを整形（tags-basedデータ使用）
  const formattedVideos: VideoData[] = uniqueVideos.map((video: any) => ({
    id: video.id,
    title: video.title,
    description: video.description,
    thumbnail_url: video.thumbnail_url,
    preview_video_url: video.preview_video_url,
    maker: video.maker,
    genre: (() => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      return tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
    })(), // tagsから動的に生成されたgenre
    price: video.price,
    sample_video_url: video.sample_video_url,
    image_urls: video.image_urls || [],
    duration_seconds: video.duration_seconds,
    performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
    tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
    all_tags: video.all_tags || [], // tags JOINから取得されたall_tags
    created_at: video.created_at,
    updated_at: video.updated_at
  }));

  // シャッフルして多様性を確保
  const shuffledVideos = formattedVideos.sort(() => Math.random() - 0.5).slice(0, limit);
  
  // 多様性スコア計算
  const diversityScore = calculateDiversityScore(shuffledVideos);

  console.log(`Returning ${shuffledVideos.length} diverse videos (diversity score: ${diversityScore})`);

  return {
    videos: shuffledVideos,
    totalCount: shuffledVideos.length,
    diversityScore
  };
}

// Latest Feed
async function getLatestFeed(supabaseClient: any, limit: number, offset: number, excludeIds: string[]) {
  const { data: videos, error } = await supabaseClient
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
      video_performers!inner(
        performers(name)
      ),
      video_tags!inner(
        tags(name)
      )
    `)
    .not('id', 'in', excludeIds)
    .order('created_at', { ascending: false })
    .range(offset, offset + limit - 1);

  if (error) {
    throw new Error(`Failed to fetch latest videos: ${error.message}`);
  }

  const formattedVideos: VideoData[] = (videos || []).map((video: any) => ({
    id: video.id,
    title: video.title,
    description: video.description,
    thumbnail_url: video.thumbnail_url,
    preview_video_url: video.preview_video_url,
    maker: video.maker,
    genre: (() => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      return tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
    })(), // tagsから動的に生成されたgenre
    price: video.price,
    sample_video_url: video.sample_video_url,
    image_urls: video.image_urls || [],
    duration_seconds: video.duration_seconds,
    performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
    tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
    all_tags: video.all_tags || [], // tags JOINから取得されたall_tags
    created_at: video.created_at,
    updated_at: video.updated_at
  }));

  return {
    videos: formattedVideos,
    totalCount: formattedVideos.length
  };
}

// Popular Feed
async function getPopularFeed(supabaseClient: any, limit: number, offset: number, excludeIds: string[]) {
  // Approximate popularity by views or likes count
  const { data: videos, error } = await supabaseClient
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
      video_performers!inner(
        performers(name)
      ),
      video_tags!inner(
        tags(name)
      )
    `)
    .not('id', 'in', excludeIds)
    .order('created_at', { ascending: false })
    .range(offset, offset + limit - 1);

  if (error) {
    throw new Error(`Failed to fetch popular videos: ${error.message}`);
  }

  const formattedVideos: VideoData[] = (videos || []).map((video: any) => ({
    id: video.id,
    title: video.title,
    description: video.description,
    thumbnail_url: video.thumbnail_url,
    preview_video_url: video.preview_video_url,
    maker: video.maker,
    genre: (() => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      return tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
    })(), // tagsから動的に生成されたgenre
    price: video.price,
    sample_video_url: video.sample_video_url,
    image_urls: video.image_urls || [],
    duration_seconds: video.duration_seconds,
    performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
    tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
    all_tags: video.all_tags || [], // tags JOINから取得されたall_tags
    created_at: video.created_at,
    updated_at: video.updated_at
  }));

  return {
    videos: formattedVideos,
    totalCount: formattedVideos.length
  };
}

// Random Feed
async function getRandomFeed(supabaseClient: any, limit: number, offset: number, excludeIds: string[]) {
  // Get random videos using PostgreSQL RANDOM()
  const { data: videos, error } = await supabaseClient
    .rpc('get_random_videos', { 
      page_limit: limit,
      exclude_video_ids: excludeIds 
    });

  if (error) {
    console.warn('Random RPC failed, falling back to regular query:', error.message);
    
    // Fallback to regular query with OFFSET randomization（tags-basedクエリ使用）
    const randomOffset = Math.floor(Math.random() * 1000);
    const { data: fallbackVideos, error: fallbackError } = await supabaseClient
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
        video_performers!inner(
          performers(name)
        ),
        video_tags!inner(
          tags(name)
        )
      `)
      .not('id', 'in', excludeIds)
      .range(randomOffset, randomOffset + limit - 1);

    if (fallbackError) {
      throw new Error(`Failed to fetch random videos: ${fallbackError.message}`);
    }
    
    const formattedVideos: VideoData[] = (fallbackVideos || []).map((video: any) => ({
      id: video.id,
      title: video.title,
      description: video.description,
      thumbnail_url: video.thumbnail_url,
      preview_video_url: video.preview_video_url,
      maker: video.maker,
      genre: (() => {
      const tags = video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [];
      return tags.find((tag: string) => tag && tag.toLowerCase().includes('genre')) || 'Unknown';
    })(), // tagsから動的に生成されたgenre
      price: video.price,
      sample_video_url: video.sample_video_url,
      image_urls: video.image_urls || [],
      duration_seconds: video.duration_seconds,
      performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
      tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
      all_tags: video.all_tags || [], // tags JOINから取得されたall_tags
      created_at: video.created_at,
      updated_at: video.updated_at
    }));

    return {
      videos: formattedVideos.sort(() => Math.random() - 0.5), // Additional shuffle
      totalCount: formattedVideos.length
    };
  }

  // RPC成功時のデータ整形（tags-basedデータ使用）
  const formattedVideos: VideoData[] = (videos || []).map((video: any) => ({
    id: video.id,
    title: video.title,
    description: video.description,
    thumbnail_url: video.thumbnail_url,
    preview_video_url: video.preview_video_url,
    maker: video.maker,
    genre: video.genre || 'Unknown', // RPC結果のgenre、Unknownを保証
    price: video.price,
    sample_video_url: video.sample_video_url,
    image_urls: video.image_urls || [],
    duration_seconds: video.duration_seconds,
    performers: video.performers || [],
    tags: video.tags || [],
    all_tags: video.all_tags || [],
    created_at: video.created_at,
    updated_at: video.updated_at
  }));

  return {
    videos: formattedVideos,
    totalCount: formattedVideos.length
  };
}

// Diversity Score Calculation (from feed_explore)
function calculateDiversityScore(videos: VideoData[]): number {
  if (videos.length === 0) return 0;
  
  const genres = new Set(videos.map(v => v.genre));
  const makers = new Set(videos.map(v => v.maker));
  const performers = new Set(videos.flatMap(v => v.performers));
  
  // 多様性スコア = (ユニーク数の平均) / 総数
  const genreDiv = genres.size / videos.length;
  const makerDiv = makers.size / videos.length;
  const performerDiv = Math.min(performers.size / videos.length, 1); // 上限1
  
  return Math.round((genreDiv + makerDiv + performerDiv) / 3 * 100) / 100;
}

// Main serve function
export default serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const startTime = performance.now();

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const {
      feed_type = 'explore',
      limit = 20,
      offset = 0,
      exclude_ids = []
    } = await req.json().catch(() => ({}));

    let result;

    switch (feed_type) {
      case 'explore':
        result = await getExploreFeed(supabaseClient, limit, offset, exclude_ids);
        break;
      case 'latest':
        result = await getLatestFeed(supabaseClient, limit, offset, exclude_ids);
        break;
      case 'popular':
        result = await getPopularFeed(supabaseClient, limit, offset, exclude_ids);
        break;
      case 'random':
        result = await getRandomFeed(supabaseClient, limit, offset, exclude_ids);
        break;
      default:
        result = await getExploreFeed(supabaseClient, limit, offset, exclude_ids);
    }

    const processingTime = performance.now() - startTime;

    return new Response(
      JSON.stringify({
        success: true,
        data: result,
        feed_type,
        processing_time_ms: processingTime,
        timestamp: new Date().toISOString()
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Feed error:', error);
    const processingTime = performance.now() - startTime;

    return new Response(
      JSON.stringify({
        error: 'Internal server error',
        processing_time_ms: processingTime
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});