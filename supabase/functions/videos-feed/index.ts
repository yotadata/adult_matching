import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey',
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    // 認証済みユーザーのIDを取得
    const authHeader = req.headers.get('Authorization')
    let userId = null;
    if (authHeader) {
      const token = authHeader.split(' ')[1];
      const { data: { user }, error: userError } = await supabase.auth.getUser(token);
      if (userError) {
        console.error('Error getting user:', userError.message);
        // エラーがあっても処理を続行し、userIdはnullのままにする
      } else {
        userId = user?.id;
      }
    }

    let decidedVideoIds: string[] = [];
    if (userId) {
      // ユーザーがすでに判定した動画のIDを取得
      const { data: decisions, error: decisionsError } = await supabase
        .from('user_video_decisions')
        .select('video_id')
        .eq('user_id', userId);

      if (decisionsError) {
        console.error('Error fetching user decisions:', decisionsError.message);
        // エラーがあっても処理を続行し、decidedVideoIdsは空のままにする
      } else {
        decidedVideoIds = decisions?.map(d => d.video_id) || [];
      }
    }

    // Count only videos with a sample video URL and not already decided by the user
    let countQuery = supabase
      .from('videos')
      .select('id', { count: 'exact', head: true })
      .not('sample_video_url', 'is', null);

    if (decidedVideoIds.length > 0) {
      countQuery = countQuery.not('id', 'in', decidedVideoIds);
    }

    const { count: totalCount, error: countError } = await countQuery;

    if (countError) {
      console.error('Error counting videos:', countError.message);
      return new Response(JSON.stringify({ error: countError.message }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      });
    }

    const limit = 20;
    // totalCountがnullの場合を考慮
    const effectiveTotalCount = totalCount || 0;
    const offset = effectiveTotalCount > limit ? Math.floor(Math.random() * (effectiveTotalCount - limit)) : 0;

    // Fetch videos with a sample URL, not already decided by the user, and return the columns needed by the frontend
    let videosQuery = supabase
      .from('videos')
      .select(`
        id,
        title,
        description,
        external_id,
        thumbnail_url,
        performers: video_performers(performers(id, name)),
        tags: video_tags(tags(id, name))
      `)
      .not('sample_video_url', 'is', null)
      .order('id');

    if (decidedVideoIds.length > 0) {
      videosQuery = videosQuery.not('id', 'in', decidedVideoIds);
    }

    const { data: videos, error } = await videosQuery.range(offset, offset + limit - 1);

    if (error) {
      console.error('Error fetching videos:', error.message);
      return new Response(JSON.stringify({ error: error.message }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      })
    }

    // APIのレスポンスを整形
    const cleanedVideos = videos.map(video => ({
      ...video,
      performers: video.performers.map((p: any) => p.performers).filter(Boolean),
      tags: video.tags.map((t: any) => t.tags).filter(Boolean),
    }));

    return new Response(JSON.stringify(cleanedVideos), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error) {
    console.error('Unexpected error:', error.message);
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    })
  }
})