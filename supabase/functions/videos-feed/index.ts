import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

const SUPABASE_URL = Deno.env.get('SUPABASE_URL') ?? ''
const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY') ?? ''
const DEFAULT_LIMIT = 20
const MAX_LIMIT = 50

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const authHeader = req.headers.get('Authorization') ?? ''
    const requestJson = await req.json().catch(() => ({}))
    const requestedLimit = typeof requestJson.limit === 'number' ? requestJson.limit : undefined
    const pageLimit = requestedLimit && requestedLimit > 0
      ? Math.min(requestedLimit, MAX_LIMIT)
      : DEFAULT_LIMIT

    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: false },
      global: authHeader ? { headers: { Authorization: authHeader } } : undefined,
    })

    if (authHeader) {
      const { data: userData, error: userError } = await supabase.auth.getUser()
      if (!userError && userData?.user) {
        const { data: recs, error: recError } = await supabase.rpc('get_videos_recommendations', {
          user_uuid: userData.user.id,
          page_limit: pageLimit,
        })
        if (!recError && recs && recs.length > 0) {
          const payload = recs.map((item: Record<string, unknown>) => ({
            ...item,
            score: typeof item?.score === 'number' ? item.score : null,
            model_version: typeof item?.model_version === 'string' ? item.model_version : null,
          }))
          return new Response(JSON.stringify(payload), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            status: 200,
          })
        }
      }
    }

    // Fallback to existing feed (randomized list)
    const { data: feed, error: feedError } = await supabase.rpc('get_videos_feed', { page_limit: pageLimit })
    if (feedError) {
      console.error('Error fetching videos via RPC:', feedError.message)
      return new Response(JSON.stringify({ error: feedError.message }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      })
    }

    const payload = (feed ?? []).map((item: Record<string, unknown>) => ({
      ...item,
      score: item?.score ?? null,
      model_version: item?.model_version ?? null,
    }))

    return new Response(JSON.stringify(payload), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error) {
    console.error('Unexpected error:', error?.message ?? error)
    return new Response(JSON.stringify({ error: error?.message ?? 'unknown error' }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    })
  }
})
