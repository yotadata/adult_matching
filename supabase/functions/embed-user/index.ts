import { buildUserFeatureVector } from "../_shared/feature-builder.ts";
import { inferUserEmbedding } from "../_shared/onnx.ts";
import { getSupabaseClient } from "../_shared/supabase-client.ts";
import { fetchUserFeatureInput } from "../_shared/user-data.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

interface RequestBody {
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

    if (!force) {
      const { data: existing } = await supabase
        .from("user_embeddings")
        .select("embedding, updated_at")
        .eq("user_id", user.id)
        .maybeSingle();
      if (existing?.embedding) {
        return new Response(JSON.stringify({
          source: "cache",
          embedding: existing.embedding,
          updatedAt: existing.updated_at,
        }), {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
    }

    const context = await fetchUserFeatureInput(supabase, user.id);
    const featureVector = await buildUserFeatureVector(context);
    const embeddingArray = await inferUserEmbedding(
      featureVector.vector,
      featureVector.artifacts.schema.input_name,
    );
    const embedding = Array.from(embeddingArray);

    await supabase
      .from("user_embeddings")
      .upsert({
        user_id: user.id,
        embedding,
      });

    return new Response(JSON.stringify({
      source: "fresh",
      embedding,
      summary: featureVector.summary,
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
