import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";
import type { DecisionRecord, UserFeatureInput, VideoMeta } from "../_shared/feature-builder.ts";

interface SupabaseDecisionRow {
  video_id: string;
  decision_type: "like" | "nope";
  created_at: string | null;
  videos: {
    id: string;
    price: number | null;
    product_released_at: string | null;
    safety_level?: number | null;
    region_codes?: string[] | null;
    video_tags?: { tags?: { name?: string | null } | null }[] | null;
    video_performers?: { performers?: { name?: string | null } | null }[] | null;
  } | null;
}

function mapVideo(row: SupabaseDecisionRow["videos"]): VideoMeta | null {
  if (!row) return null;
  const tags = (row.video_tags ?? [])
    .map((entry) => entry?.tags?.name)
    .filter((name): name is string => typeof name === "string" && name.length > 0);
  const performers = (row.video_performers ?? [])
    .map((entry) => entry?.performers?.name)
    .filter((name): name is string => typeof name === "string" && name.length > 0);
  return {
    id: row.id,
    tags,
    performers,
    price: row.price,
    productReleasedAt: row.product_released_at,
  };
}

function mapDecisionRow(row: SupabaseDecisionRow): DecisionRecord {
  return {
    decisionType: row.decision_type,
    createdAt: row.created_at,
    video: mapVideo(row.videos),
  };
}

export async function fetchUserFeatureInput(
  client: SupabaseClient,
  userId: string,
): Promise<UserFeatureInput> {
  const [{ data: profile, error: profileError }, { data: decisions, error: decisionError }] = await Promise.all([
    client
      .from("profiles")
      .select("user_id, created_at")
      .eq("user_id", userId)
      .maybeSingle(),
    client
      .from("user_video_decisions")
      .select(
        `video_id, decision_type, created_at,
         videos (
           id, price, product_released_at, safety_level, region_codes,
           video_tags ( tags ( name ) ),
           video_performers ( performers ( name ) )
         )`
      )
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(500),
  ]);

  if (profileError) throw profileError;
  if (decisionError) throw decisionError;

  const profileMeta = profile ? { createdAt: profile.created_at } : { createdAt: null };
  const decisionRecords = (decisions ?? []).map(mapDecisionRow);
  return { profile: profileMeta, decisions: decisionRecords } satisfies UserFeatureInput;
}
