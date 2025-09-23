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

interface VideoData extends VideoWithTags {
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

    // 多様性を重視した動画取得戦略（シンプルクエリ使用）
    const diversityStrategies = [
      // ジャンル分散
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
        .order('maker')
        .range(offset, offset + Math.floor(limit * 0.4)),

      // 人気度順
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
        .order('created_at', { ascending: false })
        .range(offset, offset + Math.floor(limit * 0.3)),

      // ランダム
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

    // データを整形（tags-basedデータ使用）
    const formattedVideos: VideoData[] = uniqueVideos.map((video: any) => {
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
        duration_seconds: video.duration_seconds,
        performers: video.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
        tags: tags,
        all_tags: tags, // 全タグをall_tagsとして使用
        created_at: video.created_at,
        updated_at: video.updated_at
      };
    });

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