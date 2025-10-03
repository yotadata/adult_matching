import { getSupabaseServiceRoleClient } from "../_shared/supabase-client.ts";
import { computeUserEmbedding } from "../_shared/user-embedding.ts";
import { rerankCandidates } from "../_shared/ranking.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

interface RequestBody {
  user_id: string;
  limit?: number;
  bannedTagIds?: string[];
  allowedRegions?: string[];
  excludeVideoIds?: string[];
  maxSafetyLevel?: number;
  refresh_user_embedding?: boolean;
}

function toArrayOrNull(value?: string[]): string[] | null {
  if (!value || value.length === 0) return null;
  return value;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
  }

  try {
    const body = (await req.json().catch(() => ({}))) as RequestBody;
    const userId = body.user_id;
    if (!userId) {
      return new Response(JSON.stringify({ error: "user_id is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const supabase = getSupabaseServiceRoleClient();

    const limit = Math.max(1, Math.min(body.limit ?? 40, 200));
    const refreshEmbedding = !!body.refresh_user_embedding;

    let embedding: number[] | null = null;
    let embeddingSource: "cache" | "fresh" = "cache";
    let embeddingSummary: unknown = null;

    if (!refreshEmbedding) {
      const { data: existing, error } = await supabase
        .from("user_embeddings")
        .select("embedding")
        .eq("user_id", userId)
        .maybeSingle();
      if (error) throw error;
      if (existing?.embedding) {
        embedding = existing.embedding as number[];
      }
    }

    if (!embedding) {
      const result = await computeUserEmbedding(supabase, userId, { persist: true });
      embedding = result.embedding;
      embeddingSummary = result.summary;
      embeddingSource = "fresh";
    }

    const { data: annCandidates, error: rpcError } = await supabase.rpc("recommend_videos_ann", {
      p_embedding: embedding,
      p_limit: limit,
      p_banned_tag_ids: toArrayOrNull(body.bannedTagIds),
      p_max_safety_level: typeof body.maxSafetyLevel === "number" ? body.maxSafetyLevel : 1,
      p_allowed_regions: toArrayOrNull(body.allowedRegions),
      p_exclude_video_ids: toArrayOrNull(body.excludeVideoIds),
    });

    if (rpcError) {
      throw rpcError;
    }

    const candidates = annCandidates ?? [];
    if (candidates.length === 0) {
      return new Response(JSON.stringify({
        userId,
        embeddingSource,
        items: [],
        embeddingSummary,
      }), {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const videoIds = candidates.map((entry: any) => entry.video_id);
    const { data: videoRows, error: videoError } = await supabase
      .from("videos")
      .select(
        `id, title, description, external_id, thumbnail_url, product_released_at, price, safety_level, region_codes,
         video_tags ( tags ( name ) ),
         video_performers ( performers ( name ) )`
      )
      .in("id", videoIds);

    if (videoError) throw videoError;

    const videoMap = new Map<string, any>();
    for (const row of videoRows ?? []) {
      const tags = (row.video_tags ?? [])
        .map((entry: any) => entry?.tags?.name)
        .filter((name: unknown): name is string => typeof name === "string" && name.length > 0);
      const performers = (row.video_performers ?? [])
        .map((entry: any) => entry?.performers?.name)
        .filter((name: unknown): name is string => typeof name === "string" && name.length > 0);
      videoMap.set(row.id, {
        id: row.id,
        title: row.title,
        description: row.description,
        external_id: row.external_id,
        thumbnail_url: row.thumbnail_url,
        product_released_at: row.product_released_at,
        price: row.price,
        safety_level: row.safety_level,
        region_codes: row.region_codes,
        tags,
        performers,
      });
    }

    const merged = candidates
      .map((candidate: any) => {
        const video = videoMap.get(candidate.video_id);
        if (!video) return null;
        const distance = typeof candidate.distance === "number" ? candidate.distance : 0;
        const similarity = typeof candidate.similarity === "number"
          ? candidate.similarity
          : 1 - distance;
        return {
          video,
          similarity,
          distance,
        };
      })
      .filter((entry: any) => entry);

    const ranked = rerankCandidates(merged).slice(0, limit);

    return new Response(JSON.stringify({
      userId,
      embeddingSource,
      embeddingSummary,
      items: ranked,
    }), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[ai-recommend-v5]", error);
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
