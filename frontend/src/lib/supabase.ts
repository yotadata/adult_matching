import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase URL or Anon Key. Make sure NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY are set in your environment variables.');
}

// Warn in production if running on HTTPS while Supabase URL is HTTP (blocked mixed content)
try {
  if (typeof window !== 'undefined') {
    const isHttps = window.location.protocol === 'https:';
    if (isHttps && supabaseUrl?.startsWith('http://')) {
      // Do not throw to avoid breaking the app completely, but log loudly.
      // Mixed content will block network requests and appear as "no request fired" in DevTools.
      // Fix by setting NEXT_PUBLIC_SUPABASE_URL to https.
      // eslint-disable-next-line no-console
      console.error('[Supabase] Mixed content: page over HTTPS but NEXT_PUBLIC_SUPABASE_URL is HTTP. Update env to use https://');
    }
  }
} catch {}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
