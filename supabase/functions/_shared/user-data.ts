import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";
import type { UserEmbeddingFeaturePayload } from "../_shared/feature-builder.ts";

function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
}

export async function fetchUserFeatureInput(
  client: SupabaseClient,
  userId: string,
): Promise<UserEmbeddingFeaturePayload> {
  const { data, error } = await client.rpc("get_user_embedding_features", { user_id: userId });
  if (error) throw error;

  const payload = (data ?? {}) as Record<string, unknown>;
  const decisionStats = (payload["decision_stats"] ?? {}) as Record<string, unknown>;
  const priceStats = (payload["price_stats"] ?? {}) as Record<string, unknown>;

  return {
    user_id: typeof payload["user_id"] === "string" ? (payload["user_id"] as string) : userId,
    profile: (payload["profile"] as UserEmbeddingFeaturePayload["profile"]) ?? { created_at: null },
    decision_stats: {
      like_count: decisionStats["like_count"] as number | undefined,
      nope_count: decisionStats["nope_count"] as number | undefined,
      decision_count: decisionStats["decision_count"] as number | undefined,
      recent_like_at: decisionStats["recent_like_at"] as string | undefined,
      average_like_hour: decisionStats["average_like_hour"] as number | undefined,
    },
    price_stats: {
      mean: priceStats["mean"] as number | undefined,
      median: priceStats["median"] as number | undefined,
    },
    tags: normalizeStringArray(payload["tags"]),
    performers: normalizeStringArray(payload["performers"]),
    now: new Date(),
  } satisfies UserEmbeddingFeaturePayload;
}
