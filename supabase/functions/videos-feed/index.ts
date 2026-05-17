import { createClient, type SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0"

type BookEntry = {
  id: string
  title: string | null
  description: string | null
  external_id: string | null
  thumbnail_url: string | null
  sample_image_urls: string[] | null
  author: string | null
  product_released_at: string | null
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

type EdgeSupabaseClient = SupabaseClient<any, "public", any>

const ensureArray = <T>(value: unknown): T[] => {
  if (!Array.isArray(value)) return []
  return value as T[]
}

const ensureId = (value: unknown): string | null => {
  if (typeof value === "string" && value.length > 0) return value
  if (typeof value === "number" || typeof value === "bigint") return value.toString()
  return null
}

const toStringOrNull = (value: unknown): string | null =>
  typeof value === "string" ? value : null

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
  client: EdgeSupabaseClient,
  ids: string[],
): Promise<Map<string, string | null>> {
  if (ids.length === 0) return new Map()
  const { data, error } = await client
    .from('book_embeddings')
    .select('book_id, model_version')
    .in('book_id', ids)
  if (error) {
    console.error('Failed to fetch model_version map:', error.message)
    return new Map()
  }
  const map = new Map<string, string | null>()
  const rows = ensureArray<Record<string, unknown>>(data)
  for (const row of rows) {
    const bookId = ensureId(row.book_id)
    if (!bookId) continue
    const version = toStringOrNull(row.model_version)
    map.set(bookId, version)
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
          .from('user_book_decisions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', userId)
        if (countError) {
          console.error('user_book_decisions count error:', countError.message)
        } else {
          decisionCount = count ?? 0
        }
      }
    }

    let adjustedPageLimit = pageLimit
    let adjustedExploitationRatio = EXPLOITATION_RATIO
    let adjustedPopularityRatio = POPULARITY_RATIO

    if (userId) {
      if (decisionCount <= 100) {
        adjustedPageLimit = 20
        adjustedExploitationRatio = 0.5
        adjustedPopularityRatio = 0.3
      } else {
        adjustedPageLimit = 30
        adjustedExploitationRatio = 0.7
        adjustedPopularityRatio = 0.1
      }
    } else {
      adjustedPageLimit = 20
      adjustedExploitationRatio = 0
      adjustedPopularityRatio = 0.5
    }

    const exploitationTarget = Math.max(1, Math.floor(adjustedPageLimit * adjustedExploitationRatio))
    const popularityTarget = Math.max(1, Math.floor(adjustedPageLimit * adjustedPopularityRatio))
    const explorationTarget = Math.max(0, adjustedPageLimit - exploitationTarget - popularityTarget)

    const seen = new Set<string>()
    const exploitation: BookEntry[] = []
    const popularity: BookEntry[] = []
    const exploration: BookEntry[] = []

    let exploitRawCount = 0
    let exploitError: string | null = null
    if (userId && adjustedExploitationRatio > 0) {
      const { data: recs, error: recError } = await supabase.rpc('get_books_recommendations', {
        p_user_id: userId,
        p_limit: Math.max(adjustedPageLimit * CANDIDATE_MULTIPLIER, 100),
      })
      if (recError) {
        exploitError = recError.message
        console.error('get_books_recommendations error:', recError.message)
      } else if (recs && recs.length > 0) {
        exploitRawCount = recs.length
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
            sample_image_urls: Array.isArray(item.sample_image_urls) ? item.sample_image_urls as string[] : null,
            author: (item.author ?? null) as string | null,
            product_released_at: (item.product_released_at ?? null) as string | null,
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

    let popularityRawCount = 0
    let popularityError: string | null = null
    if (popularityTarget > 0) {
      const { data: popData, error: popError } = await supabase.rpc('get_popular_books', {
        p_user_id: userId,
        p_limit: popularityTarget * CANDIDATE_MULTIPLIER,
        p_days: popularLookbackDays,
      })
      if (popError) {
        popularityError = popError.message
        console.error('get_popular_books error:', popError.message)
      } else if (popData && popData.length > 0) {
        popularityRawCount = popData.length
        const popIds = (popData as { book_id: string }[]).map((item) => item.book_id)
        const modelMap = await fetchModelVersions(supabase, popIds)

        const { data: bookDetails } = await supabase
          .from('books')
          .select('id, title, external_id, thumbnail_url, sample_image_urls, author, product_released_at')
          .in('id', popIds)
        const detailMap = new Map<string, Record<string, unknown>>()
        for (const row of (bookDetails ?? []) as Record<string, unknown>[]) {
          detailMap.set(String(row.id), row)
        }

        for (const item of popData as { book_id: string; score: number }[]) {
          const id = item.book_id
          if (seen.has(id)) continue
          const detail = detailMap.get(id) ?? {}
          popularity.push({
            id,
            title: toStringOrNull(detail.title),
            description: null,
            external_id: toStringOrNull(detail.external_id),
            thumbnail_url: toStringOrNull(detail.thumbnail_url),
            sample_image_urls: Array.isArray(detail.sample_image_urls) ? detail.sample_image_urls as string[] : null,
            author: toStringOrNull(detail.author),
            product_released_at: toStringOrNull(detail.product_released_at),
            tags: [],
            score: item.score ?? null,
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
      const { data: randomData, error: randomError } = await supabase.rpc('get_books_feed', {
        p_limit: Math.max(explorationNeeded * CANDIDATE_MULTIPLIER, explorationTarget || DEFAULT_LIMIT),
      })
      if (randomError) {
        console.error('get_books_feed (exploration) error:', randomError.message)
      } else if (randomData && randomData.length > 0) {
        const bookIds = (randomData as { book_id: string }[]).map((item) => item.book_id)
        const modelMap = await fetchModelVersions(supabase, bookIds)

        const { data: bookDetails } = await supabase
          .from('books')
          .select('id, title, external_id, thumbnail_url, sample_image_urls, author, product_released_at')
          .in('id', bookIds)
        const detailMap = new Map<string, Record<string, unknown>>()
        for (const row of (bookDetails ?? []) as Record<string, unknown>[]) {
          detailMap.set(String(row.id), row)
        }

        const shuffled = shuffle(randomData as { book_id: string }[])
        for (const item of shuffled) {
          const id = item.book_id
          if (seen.has(id)) continue
          const detail = detailMap.get(id) ?? {}
          exploration.push({
            id,
            title: toStringOrNull(detail.title),
            description: null,
            external_id: toStringOrNull(detail.external_id),
            thumbnail_url: toStringOrNull(detail.thumbnail_url),
            sample_image_urls: Array.isArray(detail.sample_image_urls) ? detail.sample_image_urls as string[] : null,
            author: toStringOrNull(detail.author),
            product_released_at: toStringOrNull(detail.product_released_at),
            tags: [],
            score: null,
            model_version: modelMap.get(id) ?? null,
            source: 'exploration',
          })
          seen.add(id)
          if (exploration.length >= explorationTarget && seen.size >= pageLimit) break
        }
      }
    }

    const final: BookEntry[] = []
    let exploitIdx = 0, popIdx = 0, exploreIdx = 0
    while (final.length < adjustedPageLimit) {
      if (exploitIdx < exploitation.length) final.push(exploitation[exploitIdx++])
      if (final.length >= adjustedPageLimit) break
      if (popIdx < popularity.length) final.push(popularity[popIdx++])
      if (final.length >= adjustedPageLimit) break
      if (exploreIdx < exploration.length) final.push(exploration[exploreIdx++])
      if (exploitIdx >= exploitation.length && popIdx >= popularity.length && exploreIdx >= exploration.length) break
    }

    if (final.length < adjustedPageLimit) {
      const remaining = [...exploitation.slice(exploitIdx), ...popularity.slice(popIdx), ...exploration.slice(exploreIdx)]
      for (const item of remaining) {
        if (final.length >= adjustedPageLimit) break
        final.push(item)
      }
    }

    const EMBED_INTERVAL = 10
    const remainder = decisionCount % EMBED_INTERVAL
    const swipes_until_next_embed = EMBED_INTERVAL - remainder || EMBED_INTERVAL

    let userStats: Record<string, unknown> | null = null
    if (userId) {
      const STATS_WINDOW = 50
      const { data: recentDecisions, error: statsError } = await supabase
        .from('user_book_decisions')
        .select('decision_type, recommendation_source, recommendation_score')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(STATS_WINDOW)
      if (statsError) {
        console.error('user stats query error:', statsError.message)
      } else if (recentDecisions && recentDecisions.length > 0) {
        const rows = recentDecisions as { decision_type: string; recommendation_source: string | null; recommendation_score: number | null }[]
        const totalLikes = rows.filter(r => r.decision_type === 'like').length
        const likeRate = (n: number, d: number) => d > 0 ? Math.round(n / d * 1000) / 1000 : 0
        userStats = { window: rows.length, like_rate: likeRate(totalLikes, rows.length) }
      }
    }

    const payload = {
      videos: final.slice(0, adjustedPageLimit).map((item) => ({
        ...item,
        params: {
          requested_limit: adjustedPageLimit,
          exploitation_ratio: adjustedExploitationRatio,
          popularity_ratio: adjustedPopularityRatio,
          exploration_ratio: Math.max(0, 1 - adjustedExploitationRatio - adjustedPopularityRatio),
          popularity_lookback_days: popularLookbackDays,
        },
      })),
      metadata: {
        swipes_until_next_embed,
        decision_count: decisionCount,
        user_stats: userStats,
        _debug: {
          exploit_raw: exploitRawCount,
          exploit_err: exploitError,
          popularity_raw: popularityRawCount,
          popularity_err: popularityError,
        },
      },
    }

    return new Response(JSON.stringify(payload), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'unknown error'
    console.error('Unexpected error:', message)
    return new Response(JSON.stringify({ error: message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    })
  }
})
