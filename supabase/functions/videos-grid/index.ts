import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0"

type VideoEntry = {
  id: string
  title: string | null
  external_id: string | null
  thumbnail_url: string | null
  thumbnail_vertical_url: string | null
  sample_video_url: string | null
  embed_url: string | null
  product_url: string | null
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
const DEFAULT_LIMIT = 30
const MAX_LIMIT = 60
const DEFAULT_POPULAR_LOOKBACK_DAYS = 7
const EXPLOITATION_RATIO = 0.6
const POPULARITY_RATIO = 0.2
const CANDIDATE_MULTIPLIER = 3

function clampLimit(limit?: number): number {
  if (!limit || limit <= 0) return DEFAULT_LIMIT
  return Math.min(limit, MAX_LIMIT)
}


// FANZAのembedURLを生成する
function toEmbedUrl(externalId: string | null): string | null {
  if (!externalId) return null
  return `https://www.dmm.co.jp/litevideo/-/part==/cid=${externalId}/`
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const authHeader = req.headers.get('Authorization') ?? ''
    const requestJson = await req.json().catch(() => ({}))
    const pageLimit = clampLimit(typeof requestJson.limit === 'number' ? requestJson.limit : undefined)
    // cursor: 既取得動画IDのセット（クライアントが送る）
    const excludeIds: string[] = Array.isArray(requestJson.exclude_ids) ? requestJson.exclude_ids : []
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
        const { count } = await supabase
          .from('user_video_decisions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', userId)
        decisionCount = count ?? 0
      }
    }

    let adjustedExploitationRatio = EXPLOITATION_RATIO
    let adjustedPopularityRatio = POPULARITY_RATIO

    if (!userId) {
      adjustedExploitationRatio = 0
      adjustedPopularityRatio = 0.5
    } else if (decisionCount <= 100) {
      adjustedExploitationRatio = 0.5
      adjustedPopularityRatio = 0.3
    } else {
      adjustedExploitationRatio = 0.7
      adjustedPopularityRatio = 0.1
    }

    const exploitationTarget = Math.max(1, Math.floor(pageLimit * adjustedExploitationRatio))
    const popularityTarget = Math.max(1, Math.floor(pageLimit * adjustedPopularityRatio))
    const explorationTarget = Math.max(0, pageLimit - exploitationTarget - popularityTarget)
    // スコアがこの値を下回る exploitation 枠をタグ動画で差し替える
    const SCORE_THRESHOLD = 0.5

    const seen = new Set<string>(excludeIds)
    const exploitation: VideoEntry[] = []
    const popularity: VideoEntry[] = []
    const exploration: VideoEntry[] = []

    // モデル exploitation → スコア閾値未満の枠をタグ動画で差し替え
    // モデル結果ゼロ（真のコールドスタート）はタグ動画で全枠埋める
    if (userId && adjustedExploitationRatio > 0) {
      const { data: recs, error: recError } = await supabase.rpc('get_videos_recommendations', {
        user_uuid: userId,
        page_limit: Math.max(pageLimit * CANDIDATE_MULTIPLIER, 100),
      })
      if (recError) {
        console.error('get_videos_recommendations error:', recError.message)
      } else {
        for (const item of (recs ?? []) as Record<string, unknown>[]) {
          const id = String(item.id)
          if (seen.has(id)) continue
          exploitation.push({
            id,
            title: (item.title ?? null) as string | null,
            external_id: (item.external_id ?? null) as string | null,
            thumbnail_url: (item.thumbnail_url ?? null) as string | null,
            thumbnail_vertical_url: (item.thumbnail_vertical_url ?? null) as string | null,
            sample_video_url: (item.sample_video_url ?? null) as string | null,
            embed_url: toEmbedUrl((item.external_id ?? null) as string | null),
            product_url: (item.product_url ?? null) as string | null,
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

        // モデル結果ゼロ（コールドスタート）の場合は全枠をタグで埋める
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
                external_id: (item.external_id ?? null) as string | null,
                thumbnail_url: (item.thumbnail_url ?? null) as string | null,
                thumbnail_vertical_url: (item.thumbnail_vertical_url ?? null) as string | null,
                sample_video_url: (item.sample_video_url ?? null) as string | null,
                embed_url: toEmbedUrl((item.external_id ?? null) as string | null),
                product_url: (item.product_url ?? null) as string | null,
                product_released_at: (item.product_released_at ?? null) as string | null,
                image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
                video_source: (item.video_source ?? null) as string | null,
                performers: (item.performers ?? []) as unknown,
                tags: (item.tags ?? []) as unknown,
                score: null,
                model_version: null,
                source: 'exploitation_tag',
              })
              seen.add(id)
              if (tagEntries.length >= needCount) break
            }
            // 低スコア枠を差し替え、残りは末尾に追加
            let tagIdx = 0
            for (const idx of lowScoreIndices) {
              if (tagIdx >= tagEntries.length) break
              seen.delete(exploitation[idx].id)
              exploitation[idx] = tagEntries[tagIdx++]
            }
            // モデル結果不足分を末尾に追加
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
      } else {
        for (const item of (popData ?? []) as Record<string, unknown>[]) {
          const id = String(item.id)
          if (seen.has(id)) continue
          popularity.push({
            id,
            title: (item.title ?? null) as string | null,
            external_id: (item.external_id ?? null) as string | null,
            thumbnail_url: (item.thumbnail_url ?? null) as string | null,
            thumbnail_vertical_url: (item.thumbnail_vertical_url ?? null) as string | null,
            sample_video_url: (item.sample_video_url ?? null) as string | null,
            embed_url: toEmbedUrl((item.external_id ?? null) as string | null),
            product_url: (item.product_url ?? null) as string | null,
            product_released_at: (item.product_released_at ?? null) as string | null,
            image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
            video_source: (item.source ?? null) as string | null,
            performers: item.performers ?? [],
            tags: item.tags ?? [],
            score: item?.score !== undefined ? Number(item.score) : null,
            model_version: null,
            source: 'popularity',
          })
          seen.add(id)
          if (popularity.length >= popularityTarget) break
        }
      }
    }

    const explorationNeeded = Math.max(
      0,
      explorationTarget + (exploitationTarget - exploitation.length) + (popularityTarget - popularity.length),
    )

    if (explorationNeeded > 0) {
      let randomData: Record<string, unknown>[] | null = null
      let randomError: { message: string } | null = null

      const { data, error } = await supabase.rpc('get_videos_feed', {
        page_limit: Math.max(explorationNeeded * CANDIDATE_MULTIPLIER, DEFAULT_LIMIT),
      })
      randomData = data
      randomError = error
      if (randomError) {
        console.error('get_videos_feed error:', randomError.message)
      } else {
        for (const item of (randomData ?? []) as Record<string, unknown>[]) {
          const id = String(item.id)
          if (seen.has(id)) continue
          exploration.push({
            id,
            title: (item.title ?? null) as string | null,
            external_id: (item.external_id ?? null) as string | null,
            thumbnail_url: (item.thumbnail_url ?? null) as string | null,
            thumbnail_vertical_url: (item.thumbnail_vertical_url ?? null) as string | null,
            sample_video_url: (item.sample_video_url ?? null) as string | null,
            embed_url: toEmbedUrl((item.external_id ?? null) as string | null),
            product_url: (item.product_url ?? null) as string | null,
            product_released_at: (item.product_released_at ?? null) as string | null,
            image_urls: Array.isArray(item.image_urls) ? item.image_urls as string[] : null,
            video_source: (item.source ?? null) as string | null,
            performers: item.performers ?? [],
            tags: item.tags ?? [],
            score: null,
            model_version: null,
            source: 'exploration',
          })
          seen.add(id)
          if (exploration.length >= explorationNeeded) break
        }
      }
    }

    // exploitation → popularity → exploration の順で交互に配置
    const final: VideoEntry[] = []
    let ei = 0, pi = 0, xi = 0
    while (final.length < pageLimit) {
      if (ei < exploitation.length) final.push(exploitation[ei++])
      if (final.length >= pageLimit) break
      if (pi < popularity.length) final.push(popularity[pi++])
      if (final.length >= pageLimit) break
      if (xi < exploration.length) final.push(exploration[xi++])
      if (ei >= exploitation.length && pi >= popularity.length && xi >= exploration.length) break
    }

    return new Response(JSON.stringify({
      videos: final.slice(0, pageLimit),
      _debug: {
        preferredTagIds_count: preferredTagIds.length,
        preferredTagIds_first: preferredTagIds[0] ?? null,
        decisionCount,
        excludeIds_count: excludeIds.length,
        exploitation_count: exploitation.length,
        popularity_count: popularity.length,
        exploration_count: exploration.length,
      },
    }), {
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
