import { createClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

type PlaylistItem = {
  video_id: string;
  position: number;
  title?: string | null;
  source_section?: string | null;
  rationale?: string | null;
  duration_minutes?: number | null;
};

const sanitizeItems = (items: unknown): PlaylistItem[] => {
  if (!Array.isArray(items)) return [];
  const sanitized: PlaylistItem[] = [];
  items.forEach((item, index) => {
    if (!item || typeof item !== "object") return;
    const videoId = typeof (item as { video_id?: unknown }).video_id === "string"
      ? (item as { video_id: string }).video_id
      : null;
    if (!videoId) return;
    const positionRaw = (item as { position?: unknown }).position;
    const position = typeof positionRaw === "number" && Number.isFinite(positionRaw)
      ? positionRaw
      : index;
    sanitized.push({
      video_id: videoId,
      position,
      title: typeof (item as { title?: unknown }).title === "string" ? (item as { title: string }).title : null,
      source_section: typeof (item as { source_section?: unknown }).source_section === "string"
        ? (item as { source_section: string }).source_section
        : null,
      rationale: typeof (item as { rationale?: unknown }).rationale === "string"
        ? (item as { rationale: string }).rationale
        : null,
      duration_minutes: typeof (item as { duration_minutes?: unknown }).duration_minutes === "number"
        ? (item as { duration_minutes: number }).duration_minutes
        : null,
    });
  });
  return sanitized;
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (!["GET", "POST"].includes(req.method)) {
    return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get("Authorization") ?? "";
    if (!authHeader) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: false },
      global: { headers: { Authorization: authHeader } },
    });

    const { data: userData, error: userError } = await supabase.auth.getUser();
    if (userError || !userData?.user?.id) {
      console.error("[ai-recommend-playlist] getUser error:", userError?.message);
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const userId = userData.user.id;

    if (req.method === "GET") {
      const { data, error } = await supabase
        .from("ai_recommend_playlists")
        .select("id, mode_id, custom_intent, items, notes, visibility, created_at, updated_at")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(50);

      if (error) {
        console.error("[ai-recommend-playlist] select error:", error.message);
        return new Response(JSON.stringify({ error: "Failed to fetch playlists" }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ playlists: data ?? [] }), {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // POST
    const payload = await req.json().catch(() => null);
    if (!payload || typeof payload !== "object") {
      return new Response(JSON.stringify({ error: "Invalid payload" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const modeId = typeof (payload as { mode_id?: unknown }).mode_id === "string"
      ? (payload as { mode_id: string }).mode_id
      : null;
    const customIntent = (payload as { custom_intent?: unknown }).custom_intent;
    const sanitizedCustomIntent = customIntent && typeof customIntent === "object" ? customIntent : {};
    const notes = typeof (payload as { notes?: unknown }).notes === "string" ? (payload as { notes: string }).notes : null;
    const visibility = typeof (payload as { visibility?: unknown }).visibility === "string"
      ? (payload as { visibility: string }).visibility
      : "private";
    const items = sanitizeItems((payload as { items?: unknown }).items);

    if (items.length === 0) {
      return new Response(JSON.stringify({ error: "items must be a non-empty array" }), {
        status: 422,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const insertPayload = {
      user_id: userId,
      mode_id: modeId,
      custom_intent: sanitizedCustomIntent,
      items,
      notes,
      visibility,
    };

    const { data, error } = await supabase
      .from("ai_recommend_playlists")
      .insert(insertPayload)
      .select("id, mode_id, custom_intent, items, notes, visibility, created_at, updated_at")
      .single();

    if (error) {
      console.error("[ai-recommend-playlist] insert error:", error.message);
      return new Response(JSON.stringify({ error: "Failed to save playlist" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({ playlist: data }), {
      status: 201,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[ai-recommend-playlist] unexpected error:", error);
    return new Response(JSON.stringify({ error: "Unexpected error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
