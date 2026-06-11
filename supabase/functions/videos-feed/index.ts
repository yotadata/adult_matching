// deno-lint-ignore-file no-explicit-any
import { createClient, SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0"

type VideoEntry = {
  id: string
  title: string | null
  description: string | null
  external_id: string | null
  thumbnail_url: string | null
  sample_video_url: string | null
  product_released_at: string | null
  image_urls: string[] | null
  video_source: string | null
  performers: unknown
  tags: unknown
  score: number | null
  model_version: string | null
  source: 'exploitation' | 'exploitation_tag' | 'popularity' | 'exploration'
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


async function fetchModelVersions(
  client: SupabaseClient<any>,
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
    const preferredTagIds: string[] = Array.isArray(requestJson.preferred_tag_ids) ? requestJson.preferred_tag_ids : []
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

    // スコアがこの値を下回る exploitation 枠をタグ動画で差し替える
    const SCORE_THRESHOLD = 0.5

    const seen = new Set<string>()
    const exploitation: VideoEntry[] = []
    const popularity: VideoEntry[] = []
    const exploration: VideoEntry[] = []

    // モデル exploitation → スコア閾値未満の枠をタグ動画で差し替え
    // モデル結果ゼロ（真のコールドスタート）はタグ動画で全枠埋める
    if (userId && adjustedExploitationRatio > 0) {
      const { data: recs, error: recError } = await supabase.rpc('get_videos_recommendations', {
        user_uuid: userId,
        page_limit: Math.max(adjustedPageLimit * CANDIDATE_MULTIPLIER, 100),
      })
      if (recError) {
        console.error('get_videos_recommendations error:', recError.message)
      } else if (recs && recs.length > 0) {
        for (const item of recs as Record<string, unknown>[]) {
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
            image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
            video_source: (item.source ?? null) as string | null,
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

      // タグが指定されている場合、スコア閾値未満の枠をタグ動画で差し替える
      if (preferredTagIds.length > 0) {
        const lowScoreIndices = exploitation
          .map((v, i) => ({ i, score: v.score }))
          .filter(({ score }) => score === null || score < SCORE_THRESHOLD)
          .map(({ i }) => i)

        const needCount = lowScoreIndices.length > 0
          ? lowScoreIndices.length
          : exploitationTarget - exploitation.length

        if (needCount > 0) {
          const { data: tagRecs, error: tagError } = await supabase.rpc('get_videos_by_tags', {
            tag_ids: preferredTagIds,
            exclude_ids: Array.from(seen),
            p_limit: Math.max(needCount * CANDIDATE_MULTIPLIER, 30),
          })
          if (tagError) {
            console.error('get_videos_by_tags error:', tagError.message)
          } else {
            const tagEntries: VideoEntry[] = []
            for (const item of (tagRecs ?? []) as Record<string, unknown>[]) {
              const id = String(item.id)
              if (seen.has(id)) continue
              tagEntries.push({
                id,
                title: (item.title ?? null) as string | null,
                description: null,
                external_id: (item.external_id ?? null) as string | null,
                thumbnail_url: (item.thumbnail_url ?? null) as string | null,
                sample_video_url: (item.sample_video_url ?? null) as string | null,
                product_released_at: (item.product_released_at ?? null) as string | null,
                image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
                video_source: (item.source ?? null) as string | null,
                performers: (item.performers ?? []) as unknown,
                tags: (item.tags ?? []) as unknown,
                score: null,
                model_version: null,
                source: 'exploitation_tag',
              })
              seen.add(id)
              if (tagEntries.length >= needCount) break
            }
            let tagIdx = 0
            for (const idx of lowScoreIndices) {
              if (tagIdx >= tagEntries.length) break
              seen.delete(exploitation[idx].id)
              exploitation[idx] = tagEntries[tagIdx++]
            }
            while (tagIdx < tagEntries.length && exploitation.length < exploitationTarget) {
              exploitation.push(tagEntries[tagIdx++])
            }
          }
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
            image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
            video_source: (item.source ?? null) as string | null,
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
            image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
            video_source: (item.source ?? null) as string | null,
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

    const EMBED_INTERVAL = 10
    const remainder = decisionCount % EMBED_INTERVAL
    const swipes_until_next_embed = EMBED_INTERVAL - remainder || EMBED_INTERVAL

    const payload = {
      videos: final.slice(0, adjustedPageLimit).map((item) => ({
        ...item,
        params: {
          requested_limit: adjustedPageLimit,
          exploitation_ratio: adjustedExploitationRatio,
          popularity_ratio: adjustedPopularityRatio,
          exploration_ratio: Math.max(0, 1 - adjustedExploitationRatio - adjustedPopularityRatio),
          popularity_lookback_days: popularLookbackDays,
          exploitation_returned: exploitation.length,
          popularity_returned: popularity.length,
          exploration_returned: exploration.length,
        },
      })),
      metadata: {
        swipes_until_next_embed: swipes_until_next_embed,
        decision_count: decisionCount,
      },
    }

    return new Response(JSON.stringify(payload), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error)
    console.error('Unexpected error:', msg)
    return new Response(JSON.stringify({ error: msg }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    })
  }
})
