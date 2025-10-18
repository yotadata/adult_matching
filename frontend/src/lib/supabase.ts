import { createClient } from '@supabase/supabase-js';

// Use internal URL when running on the server (inside Docker network)
const isServer = typeof window === 'undefined';
const publicUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const internalUrl = process.env.SUPABASE_INTERNAL_URL;
const supabaseUrl = isServer ? (internalUrl || publicUrl) : publicUrl;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase URL or Anon Key. Ensure NEXT_PUBLIC_SUPABASE_URL (and SUPABASE_INTERNAL_URL for server) and NEXT_PUBLIC_SUPABASE_ANON_KEY are set.');
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

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Expose client for debugging in the browser (dev only)
try {
  if (typeof window !== 'undefined' && process.env.NODE_ENV !== 'production') {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).supabase = supabase;
  }
} catch {}
