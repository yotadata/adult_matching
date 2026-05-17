import { createClient, type SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

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
}

type JsonTag = { id?: string; name?: string };

interface BookCandidate {
  id: string;
  title: string | null;
  description: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  sample_image_urls: string[] | null;
  author: string | null;
  product_released_at: string | null;
  tags: Array<{ id: string; name: string }>;
  score?: number | null;
  popularity_score?: number | null;
  model_version?: string | null;
  product_url?: string | null;
  affiliate_url?: string | null;
  page_count?: number | null;
  source: "personalized" | "trending" | "fresh";
}

interface SectionItem {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  sample_image_urls: string[] | null;
  product_url: string | null | undefined;
  affiliate_url: string | null | undefined;
  author: string | null;
  tags: Array<{ id: string; name: string }>;
  page_count: number | null;
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

const ensureArray = <T extends Record<string, unknown>>(data: unknown): T[] => {
  if (!Array.isArray(data)) return [];
  return data as T[];
};

const ensureId = (value: unknown): string | null => {
  if (typeof value === "string" && value.length > 0) return value;
  if (typeof value === "number" || typeof value === "bigint") return value.toString();
  return null;
};

const toStringOrNull = (value: unknown): string | null =>
  typeof value === "string" ? value : null;

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

type EdgeSupabaseClient = SupabaseClient<any, "public", any>;

const fetchPersonalized = async (
  client: EdgeSupabaseClient,
  userId: string | null,
  limit: number,
): Promise<BookCandidate[]> => {
  if (!userId) return [];
  const { data, error } = await client.rpc("get_books_recommendations", {
    p_user_id: userId,
    p_limit: Math.max(limit * 6, 120),
  });
  if (error) {
    console.error("[ai-recommend] get_books_recommendations error:", error.message);
    return [];
  }
  const rows = ensureArray<Record<string, unknown>>(data);
  return rows.map((item) => {
    const id = ensureId(item.id ?? item.external_id);
    if (!id) return null;
    return {
      id,
      title: toStringOrNull(item.title),
      description: toStringOrNull(item.description),
      external_id: toStringOrNull(item.external_id),
      thumbnail_url: toStringOrNull(item.thumbnail_url),
      sample_image_urls: Array.isArray(item.sample_image_urls) ? item.sample_image_urls as string[] : null,
      author: toStringOrNull(item.author),
      product_released_at: toStringOrNull(item.product_released_at),
      tags: normalizeTags(item.tags),
      score: typeof item.score === "number" ? item.score : null,
      model_version: typeof item.model_version === "string" ? item.model_version : null,
      popularity_score: null,
      product_url: null,
      affiliate_url: null,
      page_count: null,
      source: "personalized" as const,
    };
  }).filter((v): v is BookCandidate => v !== null);
};

const fetchTrending = async (
  client: EdgeSupabaseClient,
  limit: number,
  lookbackDays: number,
): Promise<BookCandidate[]> => {
  const { data: popData, error } = await client.rpc("get_popular_books", {
    p_limit: limit * 5,
    p_days: lookbackDays,
  });
  if (error) {
    console.error("[ai-recommend] get_popular_books error:", error.message);
    return [];
  }
  const bookIds = ((popData ?? []) as { book_id: string; score: number }[]).map(r => r.book_id);
  if (bookIds.length === 0) return [];

  const { data: bookDetails } = await client
    .from("books")
    .select("id, title, external_id, thumbnail_url, sample_image_urls, author, product_released_at")
    .in("id", bookIds);
  const detailMap = new Map<string, Record<string, unknown>>();
  for (const row of (bookDetails ?? []) as Record<string, unknown>[]) {
    detailMap.set(String(row.id), row);
  }

  return ((popData ?? []) as { book_id: string; score: number }[]).map((item) => {
    const detail = detailMap.get(item.book_id) ?? {};
    return {
      id: item.book_id,
      title: toStringOrNull(detail.title),
      description: null,
      external_id: toStringOrNull(detail.external_id),
      thumbnail_url: toStringOrNull(detail.thumbnail_url),
      sample_image_urls: Array.isArray(detail.sample_image_urls) ? detail.sample_image_urls as string[] : null,
      author: toStringOrNull(detail.author),
      product_released_at: toStringOrNull(detail.product_released_at),
      tags: [],
      score: null,
      model_version: null,
      popularity_score: item.score ?? null,
      product_url: null,
      affiliate_url: null,
      page_count: null,
      source: "trending" as const,
    };
  });
};

const fetchFresh = async (
  client: EdgeSupabaseClient,
  limit: number,
): Promise<BookCandidate[]> => {
  const { data, error } = await client
    .from("books")
    .select("id, title, description, external_id, thumbnail_url, sample_image_urls, author, product_released_at, book_tags(tags(id, name))")
    .order("product_released_at", { ascending: false })
    .limit(limit * 5);

  if (error) {
    console.error("[ai-recommend] latest books fetch error:", error.message);
    return [];
  }

  return ensureArray<Record<string, unknown>>(data).map((item) => {
    const id = ensureId(item.id);
    if (!id) return null;
    const rawTags = (item as { book_tags?: unknown }).book_tags;
    const tags: Array<{ id: string; name: string }> = [];
    if (Array.isArray(rawTags)) {
      for (const bt of rawTags) {
        const tag = (bt as { tags?: { id: string; name: string } }).tags;
        if (tag?.id && tag?.name) tags.push({ id: tag.id, name: tag.name });
      }
    }
    return {
      id,
      title: toStringOrNull(item.title),
      description: toStringOrNull(item.description),
      external_id: toStringOrNull(item.external_id),
      thumbnail_url: toStringOrNull(item.thumbnail_url),
      sample_image_urls: Array.isArray(item.sample_image_urls) ? item.sample_image_urls as string[] : null,
      author: toStringOrNull(item.author),
      product_released_at: toStringOrNull(item.product_released_at),
      tags,
      score: null,
      model_version: null,
      popularity_score: null,
      product_url: null,
      affiliate_url: null,
      page_count: null,
      source: "fresh" as const,
    };
  }).filter((v): v is BookCandidate => v !== null);
};

const hydrateBookDetails = async (
  client: EdgeSupabaseClient,
  candidates: BookCandidate[],
): Promise<Map<string, { product_url: string | null; affiliate_url: string | null; page_count: number | null }>> => {
  const ids = Array.from(new Set(candidates.map((item) => item.id)));
  if (ids.length === 0) return new Map();

  const { data, error } = await client
    .from("books")
    .select("id, product_url, affiliate_url, page_count")
    .in("id", ids);

  if (error) {
    console.error("[ai-recommend] hydrate details error:", error.message);
    return new Map();
  }

  const map = new Map<string, { product_url: string | null; affiliate_url: string | null; page_count: number | null }>();
  for (const row of ensureArray<Record<string, unknown>>(data)) {
    const id = ensureId(row.id);
    if (!id) continue;
    map.set(id, {
      product_url: toStringOrNull(row.product_url),
      affiliate_url: toStringOrNull(row.affiliate_url),
      page_count: typeof row.page_count === "number" ? row.page_count : null,
    });
  }
  return map;
};

const pickUnique = (
  candidates: BookCandidate[],
  used: Set<string>,
  limit: number,
): BookCandidate[] => {
  const picked: BookCandidate[] = [];
  for (const candidate of candidates) {
    if (used.has(candidate.id)) continue;
    picked.push(candidate);
    used.add(candidate.id);
    if (picked.length >= limit) break;
  }
  return picked;
};

const buildReason = (
  item: BookCandidate,
  summaryPrefix: string,
  promptKeywords?: string[],
): { summary: string; detail: string; highlights: string[] } => {
  const primaryTag = item.tags?.[0]?.name;
  const highlightTags = (item.tags ?? []).slice(0, 3).map((tag) => `#${tag.name}`);

  const summaryParts: string[] = [summaryPrefix];
  if (primaryTag) summaryParts.push(`「${primaryTag}」`);
  if (item.score) summaryParts.push(`スコア ${(item.score * 100).toFixed(0)}%`);
  const summary = summaryParts.join(" / ");

  const detailParts: string[] = [];
  if (highlightTags.length > 0) detailParts.push(`タグ: ${highlightTags.join(" ")}`);
  if (item.author) detailParts.push(`著者: ${item.author}`);
  if (item.popularity_score) detailParts.push(`人気指標 ${item.popularity_score.toLocaleString("ja-JP")}`);
  if (item.product_released_at) detailParts.push(`発売: ${item.product_released_at.slice(0, 10)}`);
  if (promptKeywords && promptKeywords.length > 0) detailParts.push(`入力ワード: ${promptKeywords.join(", ")}`);

  return { summary, detail: detailParts.join(" / "), highlights: highlightTags };
};

const toSectionItems = (books: BookCandidate[], summaryPrefix: string, promptKeywords?: string[]): SectionItem[] =>
  books.map((book) => ({
    id: book.id,
    title: book.title,
    thumbnail_url: book.thumbnail_url,
    sample_image_urls: book.sample_image_urls,
    product_url: book.product_url,
    affiliate_url: book.affiliate_url,
    author: book.author,
    tags: book.tags,
    page_count: book.page_count ?? null,
    metrics: {
      score: book.score,
      popularity_score: book.popularity_score,
      product_released_at: book.product_released_at,
      source: book.source,
    },
    reason: buildReason(book, summaryPrefix, promptKeywords),
  }));

const matchesKeywords = (book: BookCandidate, keywords: string[]): boolean => {
  if (keywords.length === 0) return false;
  const haystack = [
    book.title ?? "",
    book.author ?? "",
    ...(book.tags ?? []).map((tag) => tag.name ?? ""),
  ].join(" ").toLowerCase();
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
    };
  }
  const url = new URL(req.url);
  const parseCommaList = (value: string | null): string[] | undefined => {
    if (!value) return undefined;
    return value.split(",").map((item) => item.trim()).filter((item) => item.length > 0);
  };
  return {
    prompt: url.searchParams.get("prompt") ?? undefined,
    limit_per_section: url.searchParams.has("limit_per_section")
      ? Number(url.searchParams.get("limit_per_section"))
      : undefined,
    tag_ids: parseCommaList(url.searchParams.get("tag_ids")),
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

    const detailMap = await hydrateBookDetails(supabase, [...personalized, ...trending, ...fresh]);
    const enhance = (book: BookCandidate): BookCandidate => {
      const extra = detailMap.get(book.id);
      return {
        ...book,
        product_url: extra?.product_url ?? book.product_url ?? null,
        affiliate_url: extra?.affiliate_url ?? book.affiliate_url ?? null,
        page_count: extra?.page_count ?? book.page_count ?? null,
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
        rationale: "「気になる」履歴からのあなたへのおすすめ漫画です。",
        items: toSectionItems(personalizedItems, "あなた向け"),
      });
    }

    const freshItems = pickUnique(enhancedFresh, used, limitPerSection);
    if (freshItems.length > 0) {
      sections.push({
        id: "fresh-releases",
        title: "新着ピックアップ",
        rationale: "発売日が新しい順に注目漫画を並べています。",
        items: toSectionItems(freshItems, "新着"),
      });
    }

    const trendingItems = pickUnique(enhancedTrending, used, limitPerSection);
    if (trendingItems.length > 0) {
      sections.push({
        id: "trend-now",
        title: "みんなが読んでいるトレンド",
        rationale: "コミュニティ全体で人気が高まっている漫画をピックアップしました。",
        items: toSectionItems(trendingItems, "トレンド"),
      });
    }

    const hasInputs = selectedTagIds.length > 0 || promptKeywords.length > 0;
    let promptCandidates: BookCandidate[] = [];
    let promptTitle = "AIセレクト（気分未入力）";
    let promptRationale = "キーワードやタグ未選択のため、AIがあなた向けとトレンドからピックアップしました。";
    let summaryPrefix = "AIセレクト";

    if (hasInputs) {
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending, ...enhancedFresh].filter((book) => {
        const tagMatch = selectedTagIds.length === 0 || (book.tags ?? []).some((tag) => selectedTagIds.includes(tag.id));
        const keywordMatch = promptKeywords.length === 0 || matchesKeywords(book, promptKeywords);
        return tagMatch && keywordMatch;
      });
      promptTitle = selectedTagIds.length > 0 ? "タグに基づくおすすめ" : "気分キーワードとマッチ";
      promptRationale = selectedTagIds.length > 0
        ? `タグ ${selectedTagIds.length}件に基づいて抽出しました。`
        : `入力ワード: ${promptKeywords.join(", ")}`;
      summaryPrefix = selectedTagIds.length > 0 ? "タグ検索" : "気分マッチ";
    } else {
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending];
    }

    if (promptCandidates.length === 0) {
      promptCandidates = [...enhancedPersonalized, ...enhancedTrending];
    }

    const promptItems = pickUnique(promptCandidates, new Set<string>(), limitPerSection);
    if (promptItems.length > 0) {
      sections.push({
        id: "prompt-match",
        title: promptTitle,
        rationale: promptRationale,
        items: toSectionItems(promptItems, summaryPrefix, promptKeywords),
      });
    }

    if (sections.length === 0) {
      const fallback = pickUnique([...enhancedTrending, ...enhancedFresh], used, limitPerSection);
      if (fallback.length > 0) {
        sections.push({
          id: "fallback",
          title: "おすすめセット",
          rationale: "人気と新着をミックスしました。",
          items: toSectionItems(fallback, "おすすめ"),
        });
      }
    }

    return new Response(JSON.stringify({
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
      },
    }), {
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
