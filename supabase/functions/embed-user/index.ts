import { getSupabaseServiceRoleClient } from "../_shared/supabase-client.ts";
import { computeUserEmbedding } from "../_shared/user-embedding.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

interface RequestBody {
  user_id?: string;
  force?: boolean;
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
    const { force } = body as RequestBody;
    const url = new URL(req.url);
    const userId = (body as RequestBody).user_id ?? url.searchParams.get("user_id") ?? undefined;

    if (!userId) {
      return new Response(JSON.stringify({ error: "user_id is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const supabase = getSupabaseServiceRoleClient();

    if (!force) {
      const { data: existing, error } = await supabase
        .from("user_embeddings")
        .select("embedding, updated_at")
        .eq("user_id", userId)
        .maybeSingle();
      if (error) throw error;
      if (existing?.embedding) {
        return new Response(JSON.stringify({
          userId,
          source: "cache",
          embedding: existing.embedding,
          updatedAt: existing.updated_at,
        }), {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
    }

    const { embedding, summary } = await computeUserEmbedding(supabase, userId, { persist: true });

    return new Response(JSON.stringify({
      userId,
      source: "fresh",
      embedding,
      summary,
    }), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[embed-user]", error);
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
