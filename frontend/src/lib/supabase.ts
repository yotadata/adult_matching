import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Initialize with a dummy client to prevent build errors and crashes.
// This will be overwritten by the real client if env vars are present.
let supabase: SupabaseClient = createClient('https://dummy.co', 'dummy.key');

console.log('[DEBUG] supabase.ts: Script start');

try {
  const isServer = typeof window === 'undefined';
  console.log(`[DEBUG] supabase.ts: isServer = ${isServer}`);

  const publicUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const internalUrl = process.env.SUPABASE_INTERNAL_URL;
  const supabaseUrl = isServer ? (internalUrl || publicUrl) : publicUrl;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    console.error('[DEBUG] supabase.ts: CRITICAL: Supabase URL or Anon Key is missing.', { 
      supabaseUrl: supabaseUrl ? 'OK' : 'MISSING',
      supabaseAnonKey: supabaseAnonKey ? 'OK' : 'MISSING'
    });
  } else {
    console.log('[DEBUG] supabase.ts: URL and Key are present. Creating real client...');
    console.log(`[DEBUG] supabase.ts: URL = ${supabaseUrl}`);
    supabase = createClient(supabaseUrl, supabaseAnonKey);
    console.log('[DEBUG] supabase.ts: Supabase real client created successfully.');
  }

  // Warn in browser if running on HTTPS while Supabase URL is HTTP (blocked mixed content)
  try {
    if (!isServer) {
      const isHttps = window.location.protocol === 'https:';
      if (isHttps && supabaseUrl?.startsWith('http://')) {
        console.error('[Supabase] Mixed content: page over HTTPS but NEXT_PUBLIC_SUPABASE_URL is HTTP. Update env to use https://');
      }
    }
  } catch {}

  type BrowserWindow = typeof window & {
    __supabase?: typeof supabase;
    __supabaseAuthLogged?: boolean;
  };

  // Expose client for debugging in the browser (local/dev only)
  try {
   const isBrowser = typeof window !== 'undefined';
   const isDev = process.env.NODE_ENV !== 'production';
   const isLocal = isBrowser && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
   if (isBrowser && isDev && isLocal) {
     const win = window as BrowserWindow;
      win.__supabase = supabase as typeof supabase;
      if (!win.__supabaseAuthLogged) {
        console.info('[Supabase] __supabase debug client available on window.__supabase');
        win.__supabaseAuthLogged = true;
      }
    }
  } catch {}

} catch (error) {
  console.error('[DEBUG] supabase.ts: CRITICAL: Uncaught error during supabase client initialization.', error);
}

export { supabase };
