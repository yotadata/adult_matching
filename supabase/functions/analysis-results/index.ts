import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

type DecisionRow = {
  video_id: string;
  decision_type: "like" | "nope";
  created_at: string;
};

type TagRow = {
  tags?: { id: string; name: string; tag_groups?: { name: string | null; show_in_ui: boolean | null } | null } | { id: string; name: string; tag_groups?: { name: string | null; show_in_ui: boolean | null } | null }[] | null;
};

type PerformerRow = {
  performers?: { id: string; name: string } | { id: string; name: string }[] | null;
};

type VideoRow = {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url?: string | null;
  affiliate_url?: string | null;
  product_url: string | null;
  video_tags?: TagRow[] | null;
  video_performers?: PerformerRow[] | null;
};

const ensureArray = <T>(value: unknown): T[] => {
  if (!Array.isArray(value)) return [];
  return value as T[];
};

type VideoDetails = {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url?: string | null;
  product_url: string | null;
  tags: Array<{ id: string; name: string; group_name: string | null }>;
  performers: Array<{ id: string; name: string }>;
};

type VideoSummary = {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url?: string | null;
  product_url: string | null;
};

type TagStat = {
  tag_id: string;
  tag_name: string;
  tag_group_name: string | null;
  likes: number;
  nopes: number;
  lastLikedAt: string | null;
  representativeVideo: VideoSummary | null;
};

type PerformerStat = {
  performer_id: string;
  performer_name: string;
  likes: number;
  nopes: number;
  lastLikedAt: string | null;
  representativeVideo: VideoSummary | null;
};

type AnalysisResponse = {
  summary: {
    total_likes: number;
    total_nope: number;
    like_ratio: number | null;
    first_decision_at: string | null;
    latest_decision_at: string | null;
    window_days: number | null;
    sample_size: number;
  };
  top_tags: Array<{
    tag_id: string;
    tag_name: string;
    tag_group_name: string | null;
    likes: number;
    nopes: number;
    like_ratio: number | null;
    share: number | null;
    last_liked_at: string | null;
    representative_video: VideoSummary | null;
  }>;
  top_performers: Array<{
    performer_id: string;
    performer_name: string;
    likes: number;
    nopes: number;
    like_ratio: number | null;
    share: number | null;
    last_liked_at: string | null;
    representative_video: VideoSummary | null;
  }>;
  recent_decisions: Array<{
    video_id: string;
    title: string | null;
    decision_type: "like" | "nope";
    decided_at: string;
    thumbnail_url: string | null;
    thumbnail_vertical_url?: string | null;
    product_url: string | null;
    tags: Array<{ id: string; name: string }>;
    performers: Array<{ id: string; name: string }>;
  }>;
};

const DEFAULT_WINDOW_DAYS = 90;
const DEFAULT_TAG_LIMIT = 6;
const DEFAULT_PERFORMER_LIMIT = 6;
const DEFAULT_RECENT_LIMIT = 10;
const MAX_FETCH = 2000;
const PAGE_SIZE = 500;
const VIDEO_CHUNK_SIZE = 50;

const emptyResponse = (windowDays: number | null): AnalysisResponse => ({
  summary: {
    total_likes: 0,
    total_nope: 0,
    like_ratio: null,
    first_decision_at: null,
    latest_decision_at: null,
    window_days: windowDays,
    sample_size: 0,
  },
  top_tags: [],
  top_performers: [],
  recent_decisions: [],
});

const coerceNumber = (value: unknown): number | null => {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const coerceBoolean = (value: unknown, fallback = false): boolean => {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    if (["true", "1", "yes"].includes(lowered)) return true;
    if (["false", "0", "no"].includes(lowered)) return false;
  }
  return fallback;
};

const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

const toIsoString = (value: string | Date | null | undefined): string | null => {
  if (!value) return null;
  try {
    const date = value instanceof Date ? value : new Date(value);
    if (Number.isNaN(date.getTime())) return null;
    return date.toISOString();
  } catch {
    return null;
  }
};

const normalizeTagRows = (rows?: TagRow[] | null): Array<{ id: string; name: string; group_name: string | null }> => {
  if (!rows) return [];
  const result: Array<{ id: string; name: string; group_name: string | null }> = [];
  for (const row of rows) {
    if (!row || !row.tags) continue;
    const tag = row.tags;
    if (Array.isArray(tag)) {
      for (const t of tag) {
        if (t && shouldDisplayTag(t)) result.push({ id: t.id, name: t.name, group_name: t.tag_groups?.name ?? null });
      }
    } else if (tag && shouldDisplayTag(tag)) {
      result.push({ id: tag.id, name: tag.name, group_name: tag.tag_groups?.name ?? null });
    }
  }
  return result;
};

const shouldDisplayTag = (tag: { tag_groups?: { show_in_ui: boolean | null } | null }): boolean => {
  if (!tag.tag_groups) return true;
  return tag.tag_groups.show_in_ui !== false;
};

const normalizePerformerRows = (rows?: PerformerRow[] | null): Array<{ id: string; name: string }> => {
  if (!rows) return [];
  const result: Array<{ id: string; name: string }> = [];
  for (const row of rows) {
    if (!row || !row.performers) continue;
    const performer = row.performers;
    if (Array.isArray(performer)) {
      for (const p of performer) {
        if (p) result.push(p);
      }
    } else if (performer) {
      result.push(performer);
    }
  }
  return result;
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "GET" && req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get("Authorization") ?? "";
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_ANON_KEY") ?? "",
      { global: { headers: { Authorization: authHeader } } },
    );

    const url = new URL(req.url);
    let body: Record<string, unknown> = {};
    if (req.method === "POST") {
      try {
        body = await req.json();
      } catch {
        body = {};
      }
    }

    const rawWindowDays = body.window_days ?? url.searchParams.get("window_days");
    let windowDays: number | null = DEFAULT_WINDOW_DAYS;
    if (rawWindowDays === null || rawWindowDays === undefined || rawWindowDays === "") {
      windowDays = DEFAULT_WINDOW_DAYS;
    } else if (typeof rawWindowDays === "string" && rawWindowDays.toLowerCase() === "all") {
      windowDays = null;
    } else if (rawWindowDays === null) {
      windowDays = null;
    } else {
      const parsed = coerceNumber(rawWindowDays);
      windowDays = parsed && parsed > 0 ? clamp(parsed, 1, 365) : null;
    }

    const includeNope = coerceBoolean(body.include_nope ?? url.searchParams.get("include_nope"), false);
    const tagLimit = clamp(coerceNumber(body.tag_limit ?? url.searchParams.get("tag_limit")) ?? DEFAULT_TAG_LIMIT, 1, 20);
    const performerLimit = clamp(
      coerceNumber(body.performer_limit ?? url.searchParams.get("performer_limit")) ?? DEFAULT_PERFORMER_LIMIT,
      1,
      20,
    );
    const recentLimit = clamp(
      coerceNumber(body.recent_limit ?? url.searchParams.get("recent_limit")) ?? DEFAULT_RECENT_LIMIT,
      1,
      50,
    );

    const windowStartIso = windowDays !== null
      ? new Date(Date.now() - windowDays * 24 * 60 * 60 * 1000).toISOString()
      : null;

    const { data: userData, error: userError } = await supabase.auth.getUser();
    if (userError) {
      console.error("[analysis-results] getUser error:", userError.message);
      const status = userError.message?.toLowerCase().includes("jwt") ? 401 : 500;
      return new Response(
        JSON.stringify({ error: status === 401 ? "unauthorized" : "internal_error" }),
        { status, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const userId = userData?.user?.id;
    if (!userId) {
      return new Response(JSON.stringify(emptyResponse(windowDays)), {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const decisions: DecisionRow[] = [];
    let page = 0;
    while (decisions.length < MAX_FETCH) {
      const from = page * PAGE_SIZE;
      const to = from + PAGE_SIZE - 1;
      let query = supabase
        .from("user_video_decisions")
        .select("video_id, decision_type, created_at")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .range(from, to);

      if (windowStartIso) {
        query = query.gte("created_at", windowStartIso);
      }

      const { data, error } = await query;
      if (error) {
        console.error("[analysis-results] decisions fetch error:", error.message);
        return new Response(JSON.stringify({ error: "internal_error" }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      if (!data || data.length === 0) break;
      decisions.push(...data);

      if (data.length < PAGE_SIZE) break;
      page += 1;
    }

    const truncatedDecisions = decisions.slice(0, MAX_FETCH);
    const sampleSize = truncatedDecisions.length;
    const totalLikes = truncatedDecisions.filter((d) => d.decision_type === "like").length;
    const totalNope = truncatedDecisions.filter((d) => d.decision_type === "nope").length;

    const uniqueVideoIds = Array.from(new Set(truncatedDecisions.map((d) => d.video_id))).filter(Boolean);
    const videoDetails = new Map<string, VideoDetails>();

    for (let i = 0; i < uniqueVideoIds.length; i += VIDEO_CHUNK_SIZE) {
      const chunk = uniqueVideoIds.slice(i, i + VIDEO_CHUNK_SIZE);
      const { data, error } = await supabase
        .from("videos")
        .select("id, title, thumbnail_url, thumbnail_vertical_url, affiliate_url, product_url, video_tags(tags(id, name, tag_groups(name, show_in_ui))), video_performers(performers(id, name))")
        .in("id", chunk);

      if (error) {
        console.error("[analysis-results] videos fetch error:", error.message);
        return new Response(JSON.stringify({ error: "internal_error" }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      const rows = ensureArray<VideoRow>(data);
      for (const row of rows) {
        const resolvedUrl = row.affiliate_url ?? row.product_url ?? null;
        videoDetails.set(row.id, {
          id: row.id,
          title: row.title ?? null,
          thumbnail_url: row.thumbnail_url ?? null,
          thumbnail_vertical_url: row.thumbnail_vertical_url ?? null,
          product_url: resolvedUrl,
          tags: normalizeTagRows(row.video_tags),
          performers: normalizePerformerRows(row.video_performers),
        });
      }
    }

    const tagStats = new Map<string, TagStat>();
    const performerStats = new Map<string, PerformerStat>();

    for (const decision of truncatedDecisions) {
      const video = videoDetails.get(decision.video_id);
      if (!video) continue;

      const tags = video.tags;
      const performers = video.performers;

      if (decision.decision_type === "like") {
        for (const tag of tags) {
          let stat = tagStats.get(tag.id);
          if (!stat) {
            stat = {
              tag_id: tag.id,
              tag_name: tag.name,
              tag_group_name: tag.group_name ?? null,
              likes: 0,
              nopes: 0,
              lastLikedAt: null,
              representativeVideo: null,
            };
            tagStats.set(tag.id, stat);
          } else if (!stat.tag_group_name && tag.group_name) {
            stat.tag_group_name = tag.group_name;
          }
          stat.likes += 1;
          if (!stat.lastLikedAt || stat.lastLikedAt < decision.created_at) {
            stat.lastLikedAt = decision.created_at;
            stat.representativeVideo = {
              id: video.id,
              title: video.title,
              thumbnail_url: video.thumbnail_url,
              thumbnail_vertical_url: video.thumbnail_vertical_url,
              product_url: video.product_url,
            };
          }
        }

        for (const performer of performers) {
          let stat = performerStats.get(performer.id);
          if (!stat) {
            stat = {
              performer_id: performer.id,
              performer_name: performer.name,
              likes: 0,
              nopes: 0,
              lastLikedAt: null,
              representativeVideo: null,
            };
            performerStats.set(performer.id, stat);
          }
          stat.likes += 1;
          if (!stat.lastLikedAt || stat.lastLikedAt < decision.created_at) {
            stat.lastLikedAt = decision.created_at;
            stat.representativeVideo = {
              id: video.id,
              title: video.title,
              thumbnail_url: video.thumbnail_url,
              thumbnail_vertical_url: video.thumbnail_vertical_url,
              product_url: video.product_url,
            };
          }
        }
      } else if (decision.decision_type === "nope") {
        for (const tag of tags) {
          let stat = tagStats.get(tag.id);
          if (!stat && includeNope) {
            stat = {
              tag_id: tag.id,
              tag_name: tag.name,
              tag_group_name: tag.group_name ?? null,
              likes: 0,
              nopes: 0,
              lastLikedAt: null,
              representativeVideo: null,
            };
            tagStats.set(tag.id, stat);
          } else if (stat && !stat.tag_group_name && tag.group_name) {
            stat.tag_group_name = tag.group_name;
          }
          if (stat) stat.nopes += 1;
        }

        for (const performer of performers) {
          let stat = performerStats.get(performer.id);
          if (!stat && includeNope) {
            stat = {
              performer_id: performer.id,
              performer_name: performer.name,
              likes: 0,
              nopes: 0,
              lastLikedAt: null,
              representativeVideo: null,
            };
            performerStats.set(performer.id, stat);
          }
          if (stat) stat.nopes += 1;
        }
      }
    }

    const sortedTags = Array.from(tagStats.values())
      .filter((stat) => stat.likes > 0 || (includeNope && stat.nopes > 0))
      .map((stat) => ({
        tag_id: stat.tag_id,
        tag_name: stat.tag_name,
        tag_group_name: stat.tag_group_name ?? null,
        likes: stat.likes,
        nopes: stat.nopes,
        like_ratio: stat.likes + stat.nopes > 0 ? stat.likes / (stat.likes + stat.nopes) : null,
        share: totalLikes > 0 ? stat.likes / totalLikes : null,
        last_liked_at: toIsoString(stat.lastLikedAt),
        representative_video: stat.representativeVideo,
      }))
      .sort((a, b) => {
        if (b.likes !== a.likes) return b.likes - a.likes;
        const ratioA = a.like_ratio ?? 0;
        const ratioB = b.like_ratio ?? 0;
        if (ratioB !== ratioA) return ratioB - ratioA;
        return (b.nopes ?? 0) - (a.nopes ?? 0);
      })
      .slice(0, tagLimit);

    const sortedPerformers = Array.from(performerStats.values())
      .filter((stat) => stat.likes > 0 || (includeNope && stat.nopes > 0))
      .map((stat) => ({
        performer_id: stat.performer_id,
        performer_name: stat.performer_name,
        likes: stat.likes,
        nopes: stat.nopes,
        like_ratio: stat.likes + stat.nopes > 0 ? stat.likes / (stat.likes + stat.nopes) : null,
        share: totalLikes > 0 ? stat.likes / totalLikes : null,
        last_liked_at: toIsoString(stat.lastLikedAt),
        representative_video: stat.representativeVideo,
      }))
      .sort((a, b) => {
        if (b.likes !== a.likes) return b.likes - a.likes;
        const ratioA = a.like_ratio ?? 0;
        const ratioB = b.like_ratio ?? 0;
        if (ratioB !== ratioA) return ratioB - ratioA;
        return (b.nopes ?? 0) - (a.nopes ?? 0);
      })
      .slice(0, performerLimit);

    const recentDecisions = truncatedDecisions.slice(0, recentLimit).map((decision) => {
      const video = videoDetails.get(decision.video_id);
      return {
        video_id: decision.video_id,
        title: video?.title ?? null,
        decision_type: decision.decision_type,
        decided_at: toIsoString(decision.created_at) ?? new Date().toISOString(),
        thumbnail_url: video?.thumbnail_url ?? null,
        thumbnail_vertical_url: video?.thumbnail_vertical_url ?? null,
        product_url: video?.product_url ?? null,
        tags: video ? video.tags.slice(0, 6) : [],
        performers: video ? video.performers.slice(0, 6) : [],
      };
    });

    const response: AnalysisResponse = {
      summary: {
        total_likes: totalLikes,
        total_nope: totalNope,
        like_ratio: sampleSize > 0 ? totalLikes / sampleSize : null,
        first_decision_at: toIsoString(truncatedDecisions.at(-1)?.created_at ?? null),
        latest_decision_at: toIsoString(truncatedDecisions[0]?.created_at ?? null),
        window_days: windowDays,
        sample_size: sampleSize,
      },
      top_tags: sortedTags,
      top_performers: sortedPerformers,
      recent_decisions: recentDecisions,
    };

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[analysis-results] unexpected error:", message);
    return new Response(JSON.stringify({ error: "internal_error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
