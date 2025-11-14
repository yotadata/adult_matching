import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

const DEFAULT_LIMIT = 6;
const MAX_LIMIT = 12;

interface RequestPayload {
  prompt?: string;
  limit_per_section?: number;
  tag_ids?: unknown;
  performer_ids?: unknown;
}

type JsonPerformer = { id?: string; name?: string };
type JsonTag = { id?: string; name?: string };

interface VideoCandidate {
  id: string;
  title: string | null;
  description: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  sample_video_url: string | null;
  product_released_at: string | null;
  performers: Array<{ id: string; name: string }>;
  tags: Array<{ id: string; name: string }>;
  score?: number | null;
  popularity_score?: number | null;
  model_version?: string | null;
  product_url?: string | null;
  preview_video_url?: string | null;
  duration_minutes?: number | null;
  source: "personalized" | "trending" | "fresh";
}

interface SectionItem {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  product_url: string | null | undefined;
  sample_video_url: string | null;
  preview_video_url: string | null | undefined;
  tags: Array<{ id: string; name: string }>;
  performers: Array<{ id: string; name: string }>;
  duration_minutes: number | null;
  metrics: {
    score?: number | null;
    popularity_score?: number | null;
    product_released_at?: string | null;
    source: string;
  };
  reason: {
    summary: string;
    detail: string;
    highlights: string[];
  };
}

interface Section {
  id: string;
  title: string;
  rationale: string;
  items: SectionItem[];
}

const toArray = <T>(value: unknown): T[] => {
  if (!value) return [];
  if (Array.isArray(value)) return value as T[];
  return [];
};

const normalizePerformers = (value: unknown): Array<{ id: string; name: string }> => {
  const list = toArray<JsonPerformer>(value);
  return list
    .filter((item): item is { id: string; name: string } => Boolean(item?.id && item?.name))
    .map((item) => ({ id: item.id!, name: item.name! }));
};

const normalizeTags = (value: unknown): Array<{ id: string; name: string }> => {
  const list = toArray<JsonTag>(value);
  return list
    .filter((item): item is { id: string; name: string } => Boolean(item?.id && item?.name))
    .map((item) => ({ id: item.id!, name: item.name! }));
};

const extractNestedEntities = (
  rows: unknown,
  key: "tags" | "performers",
): Array<{ id: string; name: string }> => {
  if (!Array.isArray(rows)) return [];
  const result: Array<{ id: string; name: string }> = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const nested = (row as Record<string, unknown>)[key];
    if (Array.isArray(nested)) {
      for (const entity of nested) {
        if (!entity || typeof entity !== "object") continue;
        const id = (entity as { id?: unknown }).id;
        const name = (entity as { name?: unknown }).name;
        if (typeof id === "string" && typeof name === "string") {
          result.push({ id, name });
        }
      }
    } else if (nested && typeof nested === "object") {
      const id = (nested as { id?: unknown }).id;
      const name = (nested as { name?: unknown }).name;
      if (typeof id === "string" && typeof name === "string") {
        result.push({ id, name });
      }
    }
  }
  return result;
};

const clampLimit = (limit?: number | null) => {
  if (typeof limit !== "number" || Number.isNaN(limit) || limit <= 0) return DEFAULT_LIMIT;
  return Math.min(Math.floor(limit), MAX_LIMIT);
};

const sanitizePrompt = (prompt?: string | null): string => (prompt ?? "").trim().slice(0, 120);

const sanitizeIdList = (value?: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  const result: string[] = [];
  for (const item of value) {
    if (typeof item === "string" && item.trim().length > 0) {
      result.push(item);
      if (result.length >= 5) break;
    }
  }
  return result;
};

const extractKeywords = (prompt: string): string[] =>
  prompt
    .split(/[、。,\.\s]+/)
    .map((word) => word.trim().toLowerCase())
    .filter((word) => word.length > 0);

const fetchPersonalized = async (
  client: ReturnType<typeof createClient>,
  userId: string | null,
  limit: number,
): Promise<VideoCandidate[]> => {
  if (!userId) return [];
  const { data, error } = await client.rpc("get_videos_recommendations", {
    user_uuid: userId,
    page_limit: limit * 2,
  });
  if (error) {
    console.error("[ai-recommend] get_videos_recommendations error:", error.message);
    return [];
  }
  return (data ?? []).map((item: Record<string, unknown>) => ({
    id: String(item.id),
    title: (item.title ?? null) as string | null,
    description: (item.description ?? null) as string | null,
    external_id: (item.external_id ?? null) as string | null,
    thumbnail_url: (item.thumbnail_url ?? null) as string | null,
    sample_video_url: (item.sample_video_url ?? null) as string | null,
    product_released_at: (item.product_released_at ?? null) as string | null,
    performers: normalizePerformers(item.performers),
    tags: normalizeTags(item.tags),
    score: typeof item.score === "number" ? item.score : null,
    model_version: typeof item.model_version === "string" ? item.model_version : null,
    popularity_score: null,
    product_url: null,
    preview_video_url: null,
    duration_minutes: null,
    source: "personalized",
  }));
};

const fetchTrending = async (
  client: ReturnType<typeof createClient>,
  limit: number,
  lookbackDays: number,
): Promise<VideoCandidate[]> => {
  const { data, error } = await client.rpc("get_popular_videos", {
    limit_count: limit * 3,
    lookback_days: lookbackDays,
  });
  if (error) {
    console.error("[ai-recommend] get_popular_videos error:", error.message);
    return [];
  }
  return (data ?? []).map((item: Record<string, unknown>) => ({
    id: String(item.id),
    title: (item.title ?? null) as string | null,
    description: (item.description ?? null) as string | null,
    external_id: (item.external_id ?? null) as string | null,
    thumbnail_url: (item.thumbnail_url ?? null) as string | null,
    sample_video_url: (item.sample_video_url ?? null) as string | null,
    product_released_at: (item.product_released_at ?? null) as string | null,
    performers: normalizePerformers(item.performers),
    tags: normalizeTags(item.tags),
    score: null,
    model_version: null,
    popularity_score: typeof item.score === "number" ? item.score : null,
    product_url: null,
    preview_video_url: null,
    duration_minutes: null,
    source: "trending",
  }));
};

const fetchFresh = async (
  client: ReturnType<typeof createClient>,
  limit: number,
): Promise<VideoCandidate[]> => {
  const { data, error } = await client
    .from("videos")
    .select("id, title, description, external_id, thumbnail_url, sample_video_url, product_released_at, video_tags(tags(id, name)), video_performers(performers(id, name)))")
    .not("sample_video_url", "is", null)
    .order("product_released_at", { ascending: false })
    .limit(limit * 3);

  if (error) {
    console.error("[ai-recommend] latest videos fetch error:", error.message);
    return [];
  }

  return (data ?? []).map((item) => ({
    id: item.id,
    title: item.title ?? null,
    description: item.description ?? null,
    external_id: item.external_id ?? null,
    thumbnail_url: item.thumbnail_url ?? null,
    sample_video_url: item.sample_video_url ?? null,
    product_released_at: item.product_released_at ?? null,
    performers: extractNestedEntities((item as { video_performers?: unknown }).video_performers, "performers"),
    tags: extractNestedEntities((item as { video_tags?: unknown }).video_tags, "tags"),
    score: null,
    model_version: null,
    popularity_score: null,
    product_url: null,
    preview_video_url: null,
    duration_minutes: null,
    source: "fresh",
  }));
};

const hydrateVideoDetails = async (
  client: ReturnType<typeof createClient>,
  candidates: VideoCandidate[],
): Promise<Map<string, { product_url: string | null; preview_video_url: string | null; duration_minutes: number | null }>> => {
  const ids = Array.from(new Set(candidates.map((item) => item.id)));
  if (ids.length === 0) return new Map();

  const { data, error } = await client
    .from("videos")
    .select("id, product_url, preview_video_url, duration_seconds")
    .in("id", ids);

  if (error) {
    console.error("[ai-recommend] hydrate details error:", error.message);
    return new Map();
  }

  const map = new Map<string, { product_url: string | null; preview_video_url: string | null; duration_minutes: number | null }>();
  for (const row of data ?? []) {
    const durationSeconds = typeof row.duration_seconds === "number" ? row.duration_seconds : null;
    map.set(row.id, {
      product_url: row.product_url ?? null,
      preview_video_url: row.preview_video_url ?? null,
      duration_minutes: durationSeconds ? Math.max(1, Math.round(durationSeconds / 60)) : null,
    });
  }
  return map;
};

const pickUnique = (
  candidates: VideoCandidate[],
  used: Set<string>,
  limit: number,
): VideoCandidate[] => {
  const picked: VideoCandidate[] = [];
  for (const candidate of candidates) {
    if (used.has(candidate.id)) continue;
    picked.push(candidate);
    used.add(candidate.id);
    if (picked.length >= limit) break;
  }
  return picked;
};

interface ReasonContext {
  summaryPrefix: string;
  promptKeywords?: string[];
  selectionSummary?: string;
}

const buildReason = (
  item: VideoCandidate,
  ctx: ReasonContext,
): { summary: string; detail: string; highlights: string[] } => {
  const primaryTag = item.tags?.[0]?.name;
  const highlightTags = (item.tags ?? []).slice(0, 3).map((tag) => `#${tag.name}`);
  const performerName = item.performers?.[0]?.name;

  const summaryParts: string[] = [ctx.summaryPrefix];
  if (primaryTag) summaryParts.push(`「${primaryTag}」`);
  if (item.score) summaryParts.push(`スコア ${(item.score * 100).toFixed(0)}%`);
  const summary = summaryParts.join(" / ");

  const detailParts: string[] = [];
  if (highlightTags.length > 0) detailParts.push(`タグ: ${highlightTags.join(" ")}`);
  if (performerName) detailParts.push(`出演: ${performerName}`);
  if (item.popularity_score) detailParts.push(`人気指標 ${item.popularity_score.toLocaleString("ja-JP")}`);
  if (item.product_released_at) detailParts.push(`リリース: ${item.product_released_at.slice(0, 10)}`);
  if (ctx.promptKeywords && ctx.promptKeywords.length > 0) {
    detailParts.push(`入力ワード: ${ctx.promptKeywords.join(", ")}`);
  }
  if (ctx.selectionSummary) {
    detailParts.push(ctx.selectionSummary);
  }

  return {
    summary,
    detail: detailParts.join(" / "),
    highlights: highlightTags,
  };
};

const toSectionItems = (videos: VideoCandidate[], ctx: ReasonContext): SectionItem[] =>
  videos.map((video) => ({
    id: video.id,
    title: video.title,
    thumbnail_url: video.thumbnail_url,
    product_url: video.product_url,
    sample_video_url: video.sample_video_url,
    preview_video_url: video.preview_video_url,
    tags: video.tags,
    performers: video.performers,
    duration_minutes: video.duration_minutes ?? null,
    metrics: {
      score: video.score,
      popularity_score: video.popularity_score,
      product_released_at: video.product_released_at,
      source: video.source,
    },
    reason: buildReason(video, ctx),
  }));

const matchesKeywords = (video: VideoCandidate, keywords: string[]): boolean => {
  if (keywords.length === 0) return false;
  const haystack = [
    video.title ?? "",
    ...(video.tags ?? []).map((tag) => tag.name ?? ""),
    ...(video.performers ?? []).map((perf) => perf.name ?? ""),
  ]
    .join(" ")
    .toLowerCase();
  return keywords.some((keyword) => haystack.includes(keyword));
};

const parseRequestPayload = async (req: Request): Promise<RequestPayload> => {
  if (req.method === "POST") {
    const json = await req.json().catch(() => null);
    if (!json || typeof json !== "object") return {};
    const payload = json as Record<string, unknown>;
    return {
      prompt: typeof payload.prompt === "string" ? payload.prompt : undefined,
      limit_per_section: typeof payload.limit_per_section === "number" ? payload.limit_per_section : undefined,
      tag_ids: Array.isArray(payload.tag_ids) ? payload.tag_ids : undefined,
      performer_ids: Array.isArray(payload.performer_ids) ? payload.performer_ids : undefined,
    };
  }
  const url = new URL(req.url);
  const parseCommaList = (value: string | null): string[] | undefined => {
    if (!value) return undefined;
    return value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  };
  return {
    prompt: url.searchParams.get("prompt") ?? undefined,
    limit_per_section: url.searchParams.has("limit_per_section")
      ? Number(url.searchParams.get("limit_per_section"))
      : undefined,
    tag_ids: parseCommaList(url.searchParams.get("tag_ids")),
    performer_ids: parseCommaList(url.searchParams.get("performer_ids")),
  };
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (!["GET", "POST"].includes(req.method)) {
    return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
  }

  try {
    const payload = await parseRequestPayload(req);
    const limitPerSection = clampLimit(payload.limit_per_section);
    const normalizedPrompt = sanitizePrompt(payload.prompt);
    const promptKeywords = extractKeywords(normalizedPrompt);
    const selectedTagIds = sanitizeIdList(payload.tag_ids);
    const selectedPerformerIds = sanitizeIdList(payload.performer_ids);

    const authHeader = req.headers.get("Authorization") ?? "";
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: false },
      global: authHeader ? { headers: { Authorization: authHeader } } : undefined,
    });

    let userId: string | null = null;

    if (authHeader) {
      const { data: userData, error: userError } = await supabase.auth.getUser();
      if (userError) {
        console.error("[ai-recommend] getUser error:", userError.message);
      } else if (userData?.user?.id) {
        userId = userData.user.id;
      }
    }

    const [personalized, trending, fresh] = await Promise.all([
      fetchPersonalized(supabase, userId, limitPerSection),
      fetchTrending(supabase, limitPerSection, 7),
      fetchFresh(supabase, limitPerSection),
    ]);

    const detailMap = await hydrateVideoDetails(supabase, [...personalized, ...trending, ...fresh]);
    const enhance = (video: VideoCandidate): VideoCandidate => {
      const extra = detailMap.get(video.id);
      return {
        ...video,
        product_url: extra?.product_url ?? video.product_url ?? null,
        preview_video_url: extra?.preview_video_url ?? video.preview_video_url ?? null,
        duration_minutes: extra?.duration_minutes ?? video.duration_minutes ?? null,
      };
    };

    const enhancedPersonalized = personalized.map(enhance);
    const enhancedTrending = trending.map(enhance);
    const enhancedFresh = fresh.map(enhance);

    const used = new Set<string>();
    const sections: Section[] = [];

    const personalizedItems = pickUnique(enhancedPersonalized, used, limitPerSection);
    if (personalizedItems.length > 0) {
      sections.push({
        id: "for-you",
        title: "あなたに合わせた提案",
        rationale: "LIKE 履歴と埋め込み類似度から抽出した作品です。",
        items: toSectionItems(personalizedItems, { summaryPrefix: "あなた向け" }),
      });
    }

    const trendingItems = pickUnique(enhancedTrending, used, limitPerSection);
    if (trendingItems.length > 0) {
      sections.push({
        id: "trend-now",
        title: "みんなが観ているトレンド",
        rationale: "コミュニティ全体で人気が高まっている作品をピックアップしました。",
        items: toSectionItems(trendingItems, { summaryPrefix: "トレンド" }),
      });
    }

    const freshItems = pickUnique(enhancedFresh, used, limitPerSection);
    if (freshItems.length > 0) {
      sections.push({
        id: "fresh-releases",
        title: "新着ピックアップ",
        rationale: "発売日が新しい順に、注目度の高い作品を並べています。",
        items: toSectionItems(freshItems, { summaryPrefix: "新着" }),
      });
    }

    const hasSelections = selectedTagIds.length > 0 || selectedPerformerIds.length > 0;
    const selectionSummary = [
      selectedTagIds.length ? `タグ ${selectedTagIds.length}件` : null,
      selectedPerformerIds.length ? `出演者 ${selectedPerformerIds.length}件` : null,
    ]
      .filter(Boolean)
      .join(" / ");

    const matchesSelection = (video: VideoCandidate): boolean => {
      const tagMatch = selectedTagIds.length === 0 || (video.tags ?? []).some((tag) => selectedTagIds.includes(tag.id));
      const performerMatch = selectedPerformerIds.length === 0 || (video.performers ?? []).some((perf) => selectedPerformerIds.includes(perf.id));
      return tagMatch && performerMatch;
    };

    let promptCandidates: VideoCandidate[] = [];
    let promptTitle = "AIセレクト（気分未入力）";
    let promptRationale = "キーワードやプリセット未選択のため、AI があなた向けとトレンドからピックアップしました。";
    let summaryPrefix = "AIセレクト";
    let reasonSelectionSummary: string | undefined;

    if (hasSelections) {
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending, ...enhancedFresh].filter(matchesSelection);
      promptTitle = "プリセットに基づくおすすめ";
      promptRationale = selectionSummary || "選択したタグ/出演者に基づいて抽出しました。";
      summaryPrefix = "プリセット";
      reasonSelectionSummary = selectionSummary;
    } else if (promptKeywords.length > 0) {
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending, ...enhancedFresh].filter((video) => matchesKeywords(video, promptKeywords));
      promptTitle = "気分キーワードとマッチ";
      promptRationale = `入力ワード: ${promptKeywords.join(", ")}`;
      summaryPrefix = "気分マッチ";
    } else {
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending];
    }

    if (promptCandidates.length === 0) {
      promptRationale = `${promptRationale} / 条件に合う作品が少なかったためAIセレクトで補完しました。`;
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending];
      summaryPrefix = hasSelections ? "プリセット" : promptKeywords.length > 0 ? "気分マッチ" : "AIセレクト";
    }

    const promptItems = pickUnique(promptCandidates, used, limitPerSection);
    if (promptItems.length > 0) {
      sections.push({
        id: "prompt-match",
        title: promptTitle,
        rationale: promptRationale,
        items: toSectionItems(promptItems, {
          summaryPrefix,
          promptKeywords,
          selectionSummary: reasonSelectionSummary,
        }),
      });
    }

    if (sections.length === 0) {
      const fallback = pickUnique([...enhancedTrending, ...enhancedFresh], used, limitPerSection);
      if (fallback.length > 0) {
        sections.push({
          id: "fallback",
          title: "おすすめセット",
          rationale: "十分な候補が得られなかったため、人気と新着をミックスしました。",
          items: toSectionItems(fallback, { summaryPrefix: "おすすめ" }),
        });
      }
    }

    const responseBody = {
      generated_at: new Date().toISOString(),
      sections,
      metadata: {
        personalized_candidates: personalized.length,
        trending_candidates: trending.length,
        fresh_candidates: fresh.length,
        prompt_keywords: promptKeywords,
        has_user_context: Boolean(userId),
        limit_per_section: limitPerSection,
        selected_tag_ids: selectedTagIds,
        selected_performer_ids: selectedPerformerIds,
      },
    };

    return new Response(JSON.stringify(responseBody), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[ai-recommend] unexpected error:", error);
    return new Response(JSON.stringify({ error: "Unexpected error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
