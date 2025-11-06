import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

type ModeKey = "focus" | "partner" | "quick" | "explore" | "relax";
type SectionKey = "core_focus" | "trend_heat" | "fresh_start" | "community_mix";

type ModeDefinition = {
  id: ModeKey;
  label: string;
  description: string;
  sectionOrder: SectionKey[];
  rationale: string;
  tone: "energetic" | "warm" | "curious" | "calm";
  defaultIntent: CustomIntent;
};

type CustomIntent = {
  duration?: "short" | "medium" | "long";
  mood?: "sweet" | "passion" | "healing" | "curious";
  context?: "solo" | "partner" | "restricted";
};

type RequestPayload = {
  mode_id?: string;
  custom_intent?: CustomIntent;
  limit_per_section?: number;
};

type JsonPerformer = { id?: string; name?: string };
type JsonTag = { id?: string; name?: string };

type VideoCandidate = {
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
};

type SectionItem = {
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
};

type Section = {
  id: string;
  title: string;
  rationale: string;
  items: SectionItem[];
};

const MODE_DEFINITIONS: ModeDefinition[] = [
  {
    id: "focus",
    label: "集中して楽しむ",
    description: "最近のLIKE傾向をもとに、相性の良い作品をピンポイントで提案します。",
    sectionOrder: ["core_focus", "fresh_start", "trend_heat"],
    rationale: "あなたの嗜好に最もフィットするラインナップを短時間で絞り込みます。",
    tone: "energetic",
    defaultIntent: { duration: "medium", mood: "passion", context: "solo" },
  },
  {
    id: "partner",
    label: "パートナーと",
    description: "共有しやすいテーマや会話のきっかけになる作品をまとめます。",
    sectionOrder: ["community_mix", "core_focus", "trend_heat"],
    rationale: "パートナーと楽しみやすい柔らかめの作品と、話題性のある作品を組み合わせました。",
    tone: "warm",
    defaultIntent: { duration: "long", mood: "sweet", context: "partner" },
  },
  {
    id: "quick",
    label: "サクッと",
    description: "短時間で満足できるショートタイム向けの提案です。",
    sectionOrder: ["fresh_start", "core_focus", "trend_heat"],
    rationale: "尺の短い作品やテンポの良い作品を優先してピックアップしています。",
    tone: "energetic",
    defaultIntent: { duration: "short", mood: "passion", context: "solo" },
  },
  {
    id: "explore",
    label: "新しい刺激",
    description: "まだ試していないタグや出演者を中心に、探索的なラインナップを提示します。",
    sectionOrder: ["trend_heat", "community_mix", "fresh_start"],
    rationale: "似た嗜好を持つユーザーが最近評価した作品と、人気上昇中の作品を組み合わせています。",
    tone: "curious",
    defaultIntent: { duration: "medium", mood: "curious", context: "solo" },
  },
  {
    id: "relax",
    label: "落ち着いて観る",
    description: "長めに楽しめる静かめの作品を中心にピックアップします。",
    sectionOrder: ["community_mix", "core_focus", "fresh_start"],
    rationale: "リラックスした時間に合う穏やかなテンポの作品をつなげました。",
    tone: "calm",
    defaultIntent: { duration: "long", mood: "healing", context: "solo" },
  },
];

const DEFAULT_MODE = MODE_DEFINITIONS[0];
const DEFAULT_LIMIT = 6;
const MAX_LIMIT = 12;

const toArray = <T>(value: unknown): T[] => {
  if (!value) return [];
  if (Array.isArray(value)) return value as T[];
  return [];
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
    if (!nested) continue;
    if (Array.isArray(nested)) {
      for (const entity of nested) {
        if (!entity || typeof entity !== "object") continue;
        const id = (entity as { id?: unknown }).id;
        const name = (entity as { name?: unknown }).name;
        if (typeof id === "string" && typeof name === "string") {
          result.push({ id, name });
        }
      }
    } else if (typeof nested === "object") {
      const id = (nested as { id?: unknown }).id;
      const name = (nested as { name?: unknown }).name;
      if (typeof id === "string" && typeof name === "string") {
        result.push({ id, name });
      }
    }
  }
  return result;
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

const clampLimit = (limit?: number | null) => {
  if (typeof limit !== "number" || Number.isNaN(limit) || limit <= 0) return DEFAULT_LIMIT;
  return Math.min(Math.floor(limit), MAX_LIMIT);
};

const resolveMode = (modeId: string | undefined | null): ModeDefinition => {
  if (!modeId) return DEFAULT_MODE;
  const normalized = modeId.toLowerCase();
  return MODE_DEFINITIONS.find((mode) => mode.id === normalized) ?? DEFAULT_MODE;
};

const mergeIntent = (base: CustomIntent, override?: CustomIntent | null): CustomIntent => {
  if (!override) return base;
  return {
    duration: override.duration ?? base.duration,
    mood: override.mood ?? base.mood,
    context: override.context ?? base.context,
  };
};

const summarizeIntent = (intent: CustomIntent): string => {
  const parts: string[] = [];
  if (intent.duration) {
    parts.push(
      intent.duration === "short" ? "短時間" : intent.duration === "medium" ? "標準的な尺" : "じっくり観られる尺",
    );
  }
  if (intent.mood) {
    const moodMap: Record<string, string> = {
      sweet: "甘めの雰囲気",
      passion: "情熱的な雰囲気",
      healing: "落ち着いた雰囲気",
      curious: "刺激的なテーマ",
    };
    parts.push(moodMap[intent.mood] ?? intent.mood);
  }
  if (intent.context) {
    const ctxMap: Record<string, string> = {
      solo: "ひとり時間",
      partner: "パートナーと共有",
      restricted: "視聴環境に制約あり",
    };
    parts.push(ctxMap[intent.context] ?? intent.context);
  }
  return parts.length > 0 ? parts.join(" / ") : "おまかせ";
};

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

const buildReason = (
  item: VideoCandidate,
  mode: ModeDefinition,
  intentSummary: string,
): { summary: string; detail: string; highlights: string[] } => {
  const primaryTag = item.tags?.[0]?.name;
  const highlightTags = (item.tags ?? []).slice(0, 3).map((tag) => `#${tag.name}`);
  const performerName = item.performers?.[0]?.name;

  const summaryParts: string[] = [];
  summaryParts.push(mode.label);
  if (primaryTag) summaryParts.push(`「${primaryTag}」要素に注目`);
  if (item.score) summaryParts.push(`適合度 ${(item.score * 100).toFixed(0)}%`);
  const summary = summaryParts.join(" / ");

  const detailParts: string[] = [];
  detailParts.push(intentSummary);
  if (highlightTags.length > 0) {
    detailParts.push(`タグ: ${highlightTags.join(" ")}`);
  }
  if (performerName) {
    detailParts.push(`出演: ${performerName}`);
  }
  if (item.popularity_score) {
    detailParts.push(`人気指標 ${item.popularity_score.toLocaleString("ja-JP")}`);
  }
  if (item.product_released_at) {
    detailParts.push(`リリース: ${item.product_released_at.slice(0, 10)}`);
  }

  return {
    summary,
    detail: detailParts.join(" / "),
    highlights: highlightTags,
  };
};

const toSectionItems = (
  videos: VideoCandidate[],
  mode: ModeDefinition,
  intentSummary: string,
): SectionItem[] => videos.map((video) => ({
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
  reason: buildReason(video, mode, intentSummary),
}));

const buildSection = (
  key: SectionKey,
  ctx: {
    personalized: VideoCandidate[];
    trending: VideoCandidate[];
    fresh: VideoCandidate[];
    mode: ModeDefinition;
    intentSummary: string;
    limit: number;
    used: Set<string>;
  },
): Section | null => {
  switch (key) {
    case "core_focus": {
      const picked = pickUnique(ctx.personalized, ctx.used, ctx.limit);
      if (picked.length === 0) return null;
      return {
        id: "core-focus",
        title: "集中セット",
        rationale: "あなたの最近のLIKE履歴と埋め込み類似度を重視したピックアップです。",
        items: toSectionItems(picked, ctx.mode, ctx.intentSummary),
      };
    }
    case "trend_heat": {
      const picked = pickUnique(ctx.trending, ctx.used, ctx.limit);
      if (picked.length === 0) return null;
      return {
        id: "trend-heat",
        title: "トレンド速報",
        rationale: "コミュニティ全体で人気上昇中の作品を、あなたの嗜好に近い順に並べました。",
        items: toSectionItems(picked, ctx.mode, ctx.intentSummary),
      };
    }
    case "fresh_start": {
      const picked = pickUnique(ctx.fresh, ctx.used, ctx.limit);
      if (picked.length === 0) return null;
      return {
        id: "fresh-start",
        title: "新着フォローアップ",
        rationale: "ここ数日の新着作品から、視聴しやすい尺や雰囲気のものを抽出しています。",
        items: toSectionItems(picked, ctx.mode, ctx.intentSummary),
      };
    }
    case "community_mix": {
      const combined = [...ctx.personalized, ...ctx.trending];
      const picked = pickUnique(combined, ctx.used, ctx.limit);
      if (picked.length === 0) return null;
      return {
        id: "community-mix",
        title: "コミュニティセレクト",
        rationale: "似た嗜好のユーザーが高評価した作品と、あなたの履歴をバランス良くミックスしました。",
        items: toSectionItems(picked, ctx.mode, ctx.intentSummary),
      };
    }
    default:
      return null;
  }
};

const parseRequestPayload = async (req: Request): Promise<RequestPayload> => {
  if (req.method === "POST") {
    const json = await req.json().catch(() => null);
    if (!json || typeof json !== "object") return {};
    const payload = json as Record<string, unknown>;
    return {
      mode_id: typeof payload.mode_id === "string" ? payload.mode_id : undefined,
      custom_intent: typeof payload.custom_intent === "object" && payload.custom_intent !== null
        ? payload.custom_intent as CustomIntent
        : undefined,
      limit_per_section: typeof payload.limit_per_section === "number" ? payload.limit_per_section : undefined,
    };
  }
  const url = new URL(req.url);
  return {
    mode_id: url.searchParams.get("mode_id") ?? undefined,
    limit_per_section: url.searchParams.has("limit_per_section")
      ? Number(url.searchParams.get("limit_per_section"))
      : undefined,
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
    const resolvedMode = resolveMode(payload.mode_id);
    const limitPerSection = clampLimit(payload.limit_per_section);
    const intent = mergeIntent(resolvedMode.defaultIntent, payload.custom_intent ?? null);
    const intentSummary = summarizeIntent(intent);

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
      fetchTrending(supabase, limitPerSection, intent.context === "partner" ? 14 : 7),
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
    for (const key of resolvedMode.sectionOrder) {
      const section = buildSection(key, {
        personalized: enhancedPersonalized,
        trending: enhancedTrending,
        fresh: enhancedFresh,
        mode: resolvedMode,
        intentSummary,
        limit: limitPerSection,
        used,
      });
      if (section && section.items.length > 0) {
        sections.push(section);
      }
    }

    if (sections.length === 0) {
      const fallbackItems = pickUnique([...enhancedTrending, ...enhancedFresh], used, limitPerSection);
      if (fallbackItems.length > 0) {
        sections.push({
          id: "fallback",
          title: "おすすめセット",
          rationale: "コンテンツ数が少なかったため、最新と人気の作品をミックスしました。",
          items: toSectionItems(fallbackItems, resolvedMode, intentSummary),
        });
      }
    }

    const responseBody = {
      generated_at: new Date().toISOString(),
      mode: {
        id: resolvedMode.id,
        label: resolvedMode.label,
        description: resolvedMode.description,
        rationale: resolvedMode.rationale,
        custom_intent: intent,
      },
      sections,
      metadata: {
        personalized_candidates: personalized.length,
        trending_candidates: trending.length,
        fresh_candidates: fresh.length,
        has_user_context: Boolean(userId),
        limit_per_section: limitPerSection,
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
