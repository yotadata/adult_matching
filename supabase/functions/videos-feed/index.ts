import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0"

type VideoEntry = {
  id: string
  title: string | null
  description: string | null
  external_id: string | null
  thumbnail_url: string | null
  sample_video_url: string | null
  product_released_at: string | null
  performers: unknown
  tags: unknown
  score: number | null
  model_version: string | null
  source: 'exploitation' | 'popularity' | 'exploration'
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

const SUPABASE_URL = Deno.env.get('SUPABASE_URL') ?? ''
const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY') ?? ''
const DEFAULT_LIMIT = 20
const MAX_LIMIT = 50
const DEFAULT_POPULAR_LOOKBACK_DAYS = 7
const EXPLOITATION_RATIO = 0.6
const POPULARITY_RATIO = 0.2
const CANDIDATE_MULTIPLIER = 3

function clampLimit(limit?: number): number {
  if (!limit || limit <= 0) return DEFAULT_LIMIT
  return Math.min(limit, MAX_LIMIT)
}

function shuffle<T>(arr: T[]): T[] {
  const clone = [...arr]
  for (let i = clone.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[clone[i], clone[j]] = [clone[j], clone[i]]
  }
  return clone
}

async function fetchModelVersions(
  client: ReturnType<typeof createClient>,
  ids: string[],
): Promise<Map<string, string | null>> {
  if (ids.length === 0) return new Map()
  const { data, error } = await client
    .from('video_embeddings')
    .select('video_id, model_version')
    .in('video_id', ids)
  if (error) {
    console.error('Failed to fetch model_version map:', error.message)
    return new Map()
  }
  const map = new Map<string, string | null>()
  for (const row of data ?? []) {
    map.set(row.video_id, row.model_version ?? null)
  }
  return map
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const authHeader = req.headers.get('Authorization') ?? ''
    const requestJson = await req.json().catch(() => ({}))
    const pageLimit = clampLimit(typeof requestJson.limit === 'number' ? requestJson.limit : undefined)
    const popularLookbackDays = typeof requestJson.popularity_days === 'number'
      ? Math.max(1, requestJson.popularity_days)
      : DEFAULT_POPULAR_LOOKBACK_DAYS

    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: false },
      global: authHeader ? { headers: { Authorization: authHeader } } : undefined,
    })

    let userId: string | null = null
    let decisionCount = 0

    if (authHeader) {
      const { data: userData, error: userError } = await supabase.auth.getUser()
      if (!userError && userData?.user) {
        userId = userData.user.id
        const { count, error: countError } = await supabase
          .from('user_video_decisions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', userId)
        if (countError) {
          console.error('user_video_decisions count error:', countError.message)
        } else {
          decisionCount = count ?? 0
        }
      }
    }

    // Adjust ratios and limit based on user engagement
    let adjustedPageLimit = pageLimit
    let adjustedExploitationRatio = EXPLOITATION_RATIO
    let adjustedPopularityRatio = POPULARITY_RATIO

    if (userId) { // Logged-in user
      if (decisionCount <= 100) {
        adjustedPageLimit = 20
        adjustedExploitationRatio = 0.5 // Increase personalization as the user has shown significant engagement
        adjustedPopularityRatio = 0.3
      } else {
        adjustedPageLimit = 30 // More videos for highly engaged users
        adjustedExploitationRatio = 0.7 // Heavily personalized
        adjustedPopularityRatio = 0.1
      }
    } else { // Guest user
      adjustedPageLimit = 20 // Guests always get 20
      adjustedExploitationRatio = 0 // No exploitation for guests
      adjustedPopularityRatio = 0.5
    }

    const exploitationTarget = Math.max(1, Math.floor(adjustedPageLimit * adjustedExploitationRatio))
    const popularityTarget = Math.max(1, Math.floor(adjustedPageLimit * adjustedPopularityRatio))
    const explorationTarget = Math.max(0, adjustedPageLimit - exploitationTarget - popularityTarget)

    const seen = new Set<string>()
    const exploitation: VideoEntry[] = []
    const popularity: VideoEntry[] = []
    const exploration: VideoEntry[] = []

    if (userId && adjustedExploitationRatio > 0) {
      const { data: recs, error: recError } = await supabase.rpc('get_videos_recommendations', {
        user_uuid: userId,
        page_limit: Math.max(adjustedPageLimit * CANDIDATE_MULTIPLIER, 100),
      })
      if (recError) {
        console.error('get_videos_recommendations error:', recError.message)
      } else if (recs && recs.length > 0) {
        const shuffled = shuffle(recs as Record<string, unknown>[]) 
        for (const item of shuffled) {
          const id = String(item.id)
          if (seen.has(id)) continue
          exploitation.push({
            id,
            title: (item.title ?? null) as string | null,
            description: (item.description ?? null) as string | null,
            external_id: (item.external_id ?? null) as string | null,
            thumbnail_url: (item.thumbnail_url ?? null) as string | null,
            sample_video_url: (item.sample_video_url ?? null) as string | null,
            product_released_at: (item.product_released_at ?? null) as string | null,
            performers: item.performers ?? [],
            tags: item.tags ?? [],
            score: typeof item.score === 'number' ? item.score : null,
            model_version: typeof item.model_version === 'string' ? item.model_version : null,
            source: 'exploitation',
          })
          seen.add(id)
          if (exploitation.length >= exploitationTarget) break
        }
      }
    }

    if (popularityTarget > 0) {
      const { data: popData, error: popError } = await supabase.rpc('get_popular_videos', {
        user_uuid: userId,
        limit_count: popularityTarget * CANDIDATE_MULTIPLIER,
        lookback_days: popularLookbackDays,
      })
      if (popError) {
        console.error('get_popular_videos error:', popError.message)
      } else if (popData && popData.length > 0) {
        const popIds = (popData as { id: string }[]).map((item) => item.id)
        const modelMap = await fetchModelVersions(supabase, popIds)
        for (const item of popData as Record<string, unknown>[]) {
          const id = String(item.id)
          if (seen.has(id)) continue
          popularity.push({
            id,
            title: (item.title ?? null) as string | null,
            description: (item.description ?? null) as string | null,
            external_id: (item.external_id ?? null) as string | null,
            thumbnail_url: (item.thumbnail_url ?? null) as string | null,
            sample_video_url: (item.sample_video_url ?? null) as string | null,
            product_released_at: (item.product_released_at ?? null) as string | null,
            performers: item.performers ?? [],
            tags: item.tags ?? [],
            score: item?.score !== undefined ? Number(item.score) : null,
            model_version: modelMap.get(id) ?? null,
            source: 'popularity',
          })
          seen.add(id)
          if (popularity.length >= popularityTarget) break
        }
      }
    }

    const explorationNeeded = Math.max(0,
      explorationTarget + (exploitationTarget - exploitation.length) + (popularityTarget - popularity.length),
    )

    if (explorationNeeded > 0) {
      const { data: randomData, error: randomError } = await supabase.rpc('get_videos_feed', {
        page_limit: Math.max(explorationNeeded * CANDIDATE_MULTIPLIER, explorationTarget || DEFAULT_LIMIT),
      })
      if (randomError) {
        console.error('get_videos_feed (exploration) error:', randomError.message)
      } else if (randomData && randomData.length > 0) {
        const randomIds = (randomData as { id: string }[]).map((item) => item.id)
        const modelMap = await fetchModelVersions(supabase, randomIds)
        for (const item of randomData as Record<string, unknown>[]) {
          const id = String(item.id)
          if (seen.has(id)) continue
          exploration.push({
            id,
            title: (item.title ?? null) as string | null,
            description: (item.description ?? null) as string | null,
            external_id: (item.external_id ?? null) as string | null,
            thumbnail_url: (item.thumbnail_url ?? null) as string | null,
            sample_video_url: (item.sample_video_url ?? null) as string | null,
            product_released_at: (item.product_released_at ?? null) as string | null,
            performers: item.performers ?? [],
            tags: item.tags ?? [],
            score: item?.score !== undefined ? Number(item.score) : null,
            model_version: modelMap.get(id) ?? null,
            source: 'exploration',
          })
          seen.add(id)
          if (exploration.length >= explorationTarget && seen.size >= pageLimit) break
        }
      }
    }

    const final: VideoEntry[] = []
    let exploitIdx = 0
    let popIdx = 0
    let exploreIdx = 0
    while (final.length < pageLimit) {
      if (exploitIdx < exploitation.length) {
        final.push(exploitation[exploitIdx++])
      }
      if (final.length >= pageLimit) break
      if (popIdx < popularity.length) {
        final.push(popularity[popIdx++])
      }
      if (final.length >= pageLimit) break
      if (exploreIdx < exploration.length) {
        final.push(exploration[exploreIdx++])
      }
      if (exploitIdx >= exploitation.length && popIdx >= popularity.length && exploreIdx >= exploration.length) {
        break
      }
    }

    if (final.length < pageLimit) {
      const remaining = [...exploitation.slice(exploitIdx), ...popularity.slice(popIdx), ...exploration.slice(exploreIdx)]
      for (const item of remaining) {
        if (final.length >= pageLimit) break
        final.push(item)
      }
    }

    const payload = final.slice(0, pageLimit).map((item) => ({
      ...item,
      params: {
        requested_limit: pageLimit,
        exploitation_ratio: EXPLOITATION_RATIO,
        popularity_ratio: POPULARITY_RATIO,
        exploration_ratio: Math.max(0, 1 - EXPLOITATION_RATIO - POPULARITY_RATIO),
        popularity_lookback_days: popularLookbackDays,
        exploitation_returned: exploitation.length,
        popularity_returned: popularity.length,
        exploration_returned: exploration.length,
      },
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
