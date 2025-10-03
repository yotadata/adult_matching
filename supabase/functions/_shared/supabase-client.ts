import { createClient, SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";

export function getSupabaseClient(req: Request): SupabaseClient {
  const url = Deno.env.get("SUPABASE_URL");
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY");
  if (!url || !anonKey) {
    throw new Error("SUPABASE_URL / SUPABASE_ANON_KEY are not configured");
  }
  const authHeader = req.headers.get("Authorization") ?? "";
  return createClient(url, anonKey, {
    global: {
      headers: {
        Authorization: authHeader,
      },
    },
  });
}

let serviceClient: SupabaseClient | null = null;

export function getSupabaseServiceRoleClient(): SupabaseClient {
  if (serviceClient) return serviceClient;
  const url = Deno.env.get("SUPABASE_URL");
  const serviceKey =
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? Deno.env.get("SUPABASE_SERVICE_KEY");
  if (!url || !serviceKey) {
    throw new Error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured");
  }
  serviceClient = createClient(url, serviceKey, {
    auth: { persistSession: false },
    global: { headers: {} },
  });
  return serviceClient;
}
