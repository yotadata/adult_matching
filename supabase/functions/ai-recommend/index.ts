import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
};

type Video = {
  id: string; // videos.id is uuid
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

  if (req.method !== 'GET' && req.method !== 'POST') {
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

    // Trending: top 10 by likes in the past 3 calendar days (UTC)
    // Note: video_popularity_daily.d is day-truncated; filter by a day boundary to avoid dropping same-day likes
    const now = new Date();
    const utcMidnightToday = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const since = new Date(utcMidnightToday.getTime() - 2 * 24 * 60 * 60 * 1000).toISOString();

    // Use materialized view to avoid RLS on individual decisions
    const { data: popRows, error: popErr } = await supabase
      .from('video_popularity_daily')
      .select('video_id, d, likes')
      .gte('d', since)
      .limit(50000);

    if (popErr) {
      console.error('[ai-recommend] popularity view error:', popErr.message);
      return new Response(JSON.stringify({ error: popErr.message }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const freq = new Map<string, number>();
    for (const row of popRows ?? []) {
      const id = String((row as any).video_id);
      const c = Number((row as any).likes) || 0;
      freq.set(id, (freq.get(id) ?? 0) + c);
    }

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
      const rank = new Map<string, number>();
      topIds.forEach((id, idx) => rank.set(id, idx));
      trending = (videos ?? []).sort((a, b) => (rank.get(a.id) ?? 9999) - (rank.get(b.id) ?? 9999));
    } else {
      // Fallback: if no trending rows (e.g., no likes in window), return latest 10 to avoid empty UI
      trending = latestVideos ?? [];
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
