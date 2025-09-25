import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
};

type Video = {
  id: number;
  title: string;
  description: string | null;
  external_id: string | null;
  product_url: string | null;
  thumbnail_url: string | null;
  created_at?: string | null;
};

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  if (req.method !== 'GET') {
    return new Response('Method Not Allowed', { status: 405, headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get('Authorization') ?? '';
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      { global: { headers: { Authorization: authHeader } } }
    );

    // Placeholder personalized recommendations (Two-Tower model not yet implemented)
    // For now: return the latest 10 videos as a stub
    const { data: latestVideos, error: latestErr } = await supabase
      .from('videos')
      .select('id, title, description, external_id, product_url, thumbnail_url, created_at')
      .order('created_at', { ascending: false })
      .limit(10);

    if (latestErr) {
      console.error('[ai-recommend] latest videos error:', latestErr.message);
      return new Response(JSON.stringify({ error: latestErr.message }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Trending: top 10 by likes in the past 3 days
    const since = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();

    // Fetch like events in the window; group on the edge
    const { data: likeRows, error: likesErr } = await supabase
      .from('user_video_decisions')
      .select('video_id')
      .eq('decision_type', 'like')
      .gte('created_at', since)
      .limit(20000);

    if (likesErr) {
      console.error('[ai-recommend] likes query error:', likesErr.message);
      return new Response(JSON.stringify({ error: likesErr.message }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const freq = new Map<number, number>();
    for (const row of likeRows ?? []) {
      const id = Number(row.video_id);
      if (!Number.isFinite(id)) continue;
      freq.set(id, (freq.get(id) ?? 0) + 1);
    }

    // Sort by count desc and take top 10 ids
    const topIds = Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([id]) => id);

    let trending: Video[] = [];
    if (topIds.length > 0) {
      const { data: videos, error: vidsErr } = await supabase
        .from('videos')
        .select('id, title, description, external_id, product_url, thumbnail_url, created_at')
        .in('id', topIds);

      if (vidsErr) {
        console.error('[ai-recommend] videos fetch error:', vidsErr.message);
        return new Response(JSON.stringify({ error: vidsErr.message }), {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }

      // Order results to match ranking
      const rank = new Map<number, number>();
      topIds.forEach((id, idx) => rank.set(id, idx));
      trending = (videos ?? []).sort((a, b) => (rank.get(a.id) ?? 9999) - (rank.get(b.id) ?? 9999));
    }

    const resBody = {
      personalized: latestVideos ?? [], // stub for now
      trending: trending,
    };

    return new Response(JSON.stringify(resBody), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (err: any) {
    console.error('[ai-recommend] unexpected error:', err?.message || err);
    return new Response(JSON.stringify({ error: err?.message || 'unknown error' }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});

