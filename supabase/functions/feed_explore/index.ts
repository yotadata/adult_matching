import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

interface VideoData {
  id: string;
  title: string;
  description: string;
  duration_seconds: number;
  thumbnail_url: string;
  preview_video_url: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
}

serve(async (req: Request) => {
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

    const { limit = 20, offset = 0 } = await req.json();

    console.log(`Fetching ${limit} diverse videos with offset ${offset}`);

    // 多様性を重視した動画取得戦略
    const diversityStrategies = [
      // ジャンル分散
      () => supabaseClient
        .from('videos')
        .select(`
          id,
          title,
          description,
          duration_seconds,
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
        .order('genre')
        .range(offset, offset + Math.floor(limit * 0.4)),

      // 人気度順
      () => supabaseClient
        .from('videos')
        .select(`
          id,
          title,
          description,
          duration_seconds,
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
        .order('published_at', { ascending: false })
        .range(offset, offset + Math.floor(limit * 0.3)),

      // ランダム
      () => supabaseClient
        .from('videos')
        .select(`
          id,
          title,
          description,
          duration_seconds,
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
      return new Response(
        JSON.stringify({ error: 'No videos available' }),
        {
          status: 404,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // 重複除去
    const uniqueVideos = allVideos.reduce((acc: any[], current: any) => {
      const isDuplicate = acc.some(video => video.id === current.id);
      if (!isDuplicate) {
        acc.push(current);
      }
      return acc;
    }, []);

    // データを整形
    const formattedVideos: VideoData[] = uniqueVideos.map((video: any) => ({
      id: video.id,
      title: video.title,
      description: video.description,
      duration_seconds: video.duration_seconds,
      thumbnail_url: video.thumbnail_url,
      preview_video_url: video.preview_video_url,
      maker: video.maker,
      genre: video.genre,
      price: video.price,
      sample_video_url: video.sample_video_url,
      image_urls: video.image_urls || [],
      performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
      tags: video.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
    }));

    // シャッフルして多様性を確保
    const shuffledVideos = formattedVideos.sort(() => Math.random() - 0.5).slice(0, limit);

    console.log(`Returning ${shuffledVideos.length} diverse videos`);

    return new Response(
      JSON.stringify({
        videos: shuffledVideos,
        total_count: shuffledVideos.length,
        offset: offset,
        limit: limit,
        diversity_score: calculateDiversityScore(shuffledVideos),
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