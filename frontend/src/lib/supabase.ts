import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Augment the global type to include our supabase client for singleton pattern
declare global {
  var __supabase: SupabaseClient | undefined;
}

const createSupabaseClient = () => {
  console.log('[DEBUG] createSupabaseClient: Called to create a new instance.');

  const isServer = typeof window === 'undefined';
  const publicUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const internalUrl = process.env.SUPABASE_INTERNAL_URL;
  const supabaseUrl = isServer ? (internalUrl || publicUrl) : publicUrl;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    console.error('[DEBUG] supabase.ts: CRITICAL: Supabase URL or Anon Key is missing.');
    // Return a dummy client to avoid crashes, but it will fail on any request.
    return createClient('https://dummy.co', 'dummy.key', { 
      auth: { persistSession: false },
      global: { fetch: async () => new Response(JSON.stringify({ error: 'Supabase client not initialized' }), { status: 500 }) }
    });
  }

  console.log(`[DEBUG] supabase.ts: Creating new client for URL: ${supabaseUrl}`);
  return createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
      autoRefreshToken: false,
    },
  });
};

// In development, use a global variable to preserve the client across HMR.
// In production, always create a new client.
const supabase = process.env.NODE_ENV === 'development'
  ? (globalThis.__supabase = globalThis.__supabase ?? createSupabaseClient())
  : createSupabaseClient();

export { supabase };
