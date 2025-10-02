import { buildUserFeatureVector } from "../_shared/feature-builder.ts";
import { inferUserEmbedding } from "../_shared/onnx.ts";
import { rerankCandidates } from "../_shared/ranking.ts";
import { getSupabaseClient } from "../_shared/supabase-client.ts";
import { fetchUserFeatureInput } from "../_shared/user-data.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

interface RequestBody {
  limit?: number;
  bannedTagIds?: string[];
  maxSafetyLevel?: number;
  allowedRegions?: string[];
  excludeVideoIds?: string[];
  forceEmbeddingRefresh?: boolean;
}

function toArrayOrNull(value?: string[]): string[] | null {
  if (!value || value.length === 0) return null;
  return value;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "GET" && req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
  }

  try {
    const body = req.method === "POST" ? await req.json().catch(() => ({})) : {};
    const { limit, bannedTagIds, maxSafetyLevel, allowedRegions, excludeVideoIds, forceEmbeddingRefresh } =
      body as RequestBody;

    const supabase = getSupabaseClient(req);
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError) throw authError;
    if (!user) {
      return new Response(JSON.stringify({ error: "unauthorized" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    let embedding: number[] | null = null;
    let embeddingSource: "cache" | "fresh" = "cache";

    if (!forceEmbeddingRefresh) {
      const { data: existing } = await supabase
        .from("user_embeddings")
        .select("embedding")
        .eq("user_id", user.id)
        .maybeSingle();
      if (existing?.embedding) {
        embedding = existing.embedding as number[];
      }
    }

    if (!embedding) {
      const context = await fetchUserFeatureInput(supabase, user.id);
      const features = await buildUserFeatureVector(context);
      const embeddingArray = await inferUserEmbedding(
        features.vector,
        features.artifacts.schema.input_name,
      );
      embedding = Array.from(embeddingArray);
      embeddingSource = "fresh";

      await supabase
        .from("user_embeddings")
        .upsert({
          user_id: user.id,
          embedding,
        });
    }

    const rpcLimit = typeof limit === "number" ? limit : 40;

    const { data: annCandidates, error: rpcError } = await supabase.rpc("recommend_videos_ann", {
      p_embedding: embedding,
      p_limit: rpcLimit,
      p_banned_tag_ids: toArrayOrNull(bannedTagIds),
      p_max_safety_level: typeof maxSafetyLevel === "number" ? maxSafetyLevel : 1,
      p_allowed_regions: toArrayOrNull(allowedRegions),
      p_exclude_video_ids: toArrayOrNull(excludeVideoIds),
    });

    if (rpcError) throw rpcError;
    const candidates = annCandidates ?? [];
    if (candidates.length === 0) {
      return new Response(JSON.stringify({
        embeddingSource,
        items: [],
      }), {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const videoIds = candidates.map((entry) => entry.video_id);
    const { data: videoRows, error: videoError } = await supabase
      .from("videos")
      .select(
        `id, title, description, external_id, thumbnail_url, product_released_at, price, safety_level, region_codes,
         video_tags ( tags ( name ) ),
         video_performers ( performers ( name ) )`
      )
      .in("id", videoIds);

    if (videoError) throw videoError;

    const videoMap = new Map(
      (videoRows ?? []).map((row) => {
        const tags = (row.video_tags ?? [])
          .map((entry) => entry?.tags?.name)
          .filter((name): name is string => typeof name === "string" && name.length > 0);
        const performers = (row.video_performers ?? [])
          .map((entry) => entry?.performers?.name)
          .filter((name): name is string => typeof name === "string" && name.length > 0);
        return [row.id, {
          ...row,
          tags,
          performers,
        }];
      }),
    );

    const merged = candidates
      .map((candidate) => {
        const video = videoMap.get(candidate.video_id);
        if (!video) return null;
        return {
          video,
          similarity: candidate.similarity ?? (candidate.distance != null ? 1 - candidate.distance : 0),
          distance: candidate.distance ?? 0,
        };
      })
      .filter((entry): entry is NonNullable<typeof entry> => !!entry);

    const ranked = rerankCandidates(merged).slice(0, rpcLimit);

    return new Response(JSON.stringify({
      embeddingSource,
      items: ranked,
    }), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[recommend]", error);
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
