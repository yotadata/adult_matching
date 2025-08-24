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

    // Count only videos with a sample video URL
    const { count: totalCount, error: countError } = await supabase
      .from('videos')
      .select('id', { count: 'exact', head: true })
      .not('sample_video_url', 'is', null);

    if (countError) {
      console.error('Error counting videos:', countError.message);
      return new Response(JSON.stringify({ error: countError.message }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      });
    }

    const limit = 20;
    const offset = totalCount && totalCount > limit ? Math.floor(Math.random() * (totalCount - limit)) : 0;

    // Fetch videos with a sample URL and return the columns needed by the frontend
    const { data: videos, error } = await supabase
      .from('videos')
      .select(`
        id,
        title,
        external_id,
        thumbnail_url,
        performers: video_performers(performers(name)),
        tags: video_tags(tags(name))
      `)
      .not('sample_video_url', 'is', null)
      .order('id')
      .range(offset, offset + limit - 1);

    if (error) {
      console.error('Error fetching videos:', error.message);
      return new Response(JSON.stringify({ error: error.message }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      })
    }

    return new Response(JSON.stringify(videos), {
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
