import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
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

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
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

    // 多様性を確保するため、異なるジャンル・メーカー・出演者から動画を取得
    const { data: videos, error } = await supabaseClient
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
      .range(offset, offset + limit - 1);

    if (error) {
      console.error('Database error:', error);
      return new Response(
        JSON.stringify({ error: 'Failed to fetch videos' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // データを整形
    const formattedVideos: VideoData[] = videos.map((video: any) => ({
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

    // 多様性を確保するため、ジャンル・メーカー・出演者でシャッフル
    const diverseVideos = ensureDiversity(formattedVideos, limit);

    return new Response(
      JSON.stringify({
        videos: diverseVideos,
        total_count: diverseVideos.length,
        has_more: diverseVideos.length === limit
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

// 多様性を確保する関数
function ensureDiversity(videos: VideoData[], targetCount: number): VideoData[] {
  if (videos.length <= targetCount) {
    return videos;
  }

  const diverseVideos: VideoData[] = [];
  const usedMakers = new Set<string>();
  const usedGenres = new Set<string>();
  const usedPerformers = new Set<string>();
  
  // まず、異なるメーカー・ジャンル・出演者から優先的に選択
  for (const video of videos) {
    if (diverseVideos.length >= targetCount) break;
    
    const hasNewMaker = video.maker && !usedMakers.has(video.maker);
    const hasNewGenre = video.genre && !usedGenres.has(video.genre);
    const hasNewPerformer = video.performers.some(p => !usedPerformers.has(p));
    
    if (hasNewMaker || hasNewGenre || hasNewPerformer) {
      diverseVideos.push(video);
      if (video.maker) usedMakers.add(video.maker);
      if (video.genre) usedGenres.add(video.genre);
      video.performers.forEach(p => usedPerformers.add(p));
    }
  }
  
  // 残りの枠を埋める
  for (const video of videos) {
    if (diverseVideos.length >= targetCount) break;
    if (!diverseVideos.includes(video)) {
      diverseVideos.push(video);
    }
  }
  
  return diverseVideos.slice(0, targetCount);
}