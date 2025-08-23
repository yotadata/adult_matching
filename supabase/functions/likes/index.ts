import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
};

interface LikeResponse {
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
  liked_at: string;
  purchased: boolean;
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

    switch (req.method) {
      case 'GET':
        return await handleGetLikes(supabaseClient, user.id);
      case 'POST':
        return await handleAddLike(supabaseClient, user.id, req);
      case 'DELETE':
        return await handleRemoveLike(supabaseClient, user.id, req);
      default:
        return new Response(
          JSON.stringify({ error: 'Method not allowed' }),
          {
            status: 405,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
    }

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

// いいね一覧取得
async function handleGetLikes(supabaseClient: any, userId: string) {
  const { limit = 50, offset = 0 } = {};

  try {
    const { data: likes, error } = await supabaseClient
      .from('likes')
      .select(`
        purchased,
        created_at,
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
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (error) {
      console.error('Database error:', error);
      return new Response(
        JSON.stringify({ error: 'Failed to fetch liked videos' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // データを整形
    const formattedLikes: LikeResponse[] = likes.map((like: any) => ({
      id: like.videos.id,
      title: like.videos.title,
      description: like.videos.description,
      thumbnail_url: like.videos.thumbnail_url,
      preview_video_url: like.videos.preview_video_url,
      maker: like.videos.maker,
      genre: like.videos.genre,
      price: like.videos.price,
      sample_video_url: like.videos.sample_video_url,
      image_urls: like.videos.image_urls || [],
      performers: like.videos.video_performers?.map((vp: any) => vp.performers?.name).filter(Boolean) || [],
      tags: like.videos.video_tags?.map((vt: any) => vt.tags?.name).filter(Boolean) || [],
      liked_at: like.created_at,
      purchased: like.purchased,
    }));

    return new Response(
      JSON.stringify({
        likes: formattedLikes,
        total_count: formattedLikes.length,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Get likes error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to get likes' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
}

// いいね追加
async function handleAddLike(supabaseClient: any, userId: string, req: Request) {
  try {
    const { video_id } = await req.json();

    if (!video_id) {
      return new Response(
        JSON.stringify({ error: 'video_id is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // 動画が存在するか確認
    const { data: video, error: videoError } = await supabaseClient
      .from('videos')
      .select('id')
      .eq('id', video_id)
      .single();

    if (videoError || !video) {
      return new Response(
        JSON.stringify({ error: 'Video not found' }),
        {
          status: 404,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // いいねを追加（既存の場合は無視）
    const { error: likeError } = await supabaseClient
      .from('likes')
      .upsert({
        user_id: userId,
        video_id: video_id,
        purchased: false,
      });

    if (likeError) {
      console.error('Like insert error:', likeError);
      return new Response(
        JSON.stringify({ error: 'Failed to add like' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    return new Response(
      JSON.stringify({ success: true, message: 'Like added successfully' }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Add like error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process like' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
}

// いいね削除
async function handleRemoveLike(supabaseClient: any, userId: string, req: Request) {
  try {
    const { video_id } = await req.json();

    if (!video_id) {
      return new Response(
        JSON.stringify({ error: 'video_id is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // いいねを削除
    const { error: deleteError } = await supabaseClient
      .from('likes')
      .delete()
      .eq('user_id', userId)
      .eq('video_id', video_id);

    if (deleteError) {
      console.error('Like delete error:', deleteError);
      return new Response(
        JSON.stringify({ error: 'Failed to remove like' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    return new Response(
      JSON.stringify({ success: true, message: 'Like removed successfully' }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );

  } catch (error) {
    console.error('Remove like error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process unlike' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
}