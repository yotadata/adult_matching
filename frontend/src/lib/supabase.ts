import { createClient } from '@supabase/supabase-js';

const ensureHttps = (url?: string) => {
  if (!url) return undefined;
  if (url.startsWith('https://')) return url;
  return `https://${url}`;
};

// Use internal URL when running on the server (inside Docker network)
const isServer = typeof window === 'undefined';
const publicUrl = ensureHttps(process.env.NEXT_PUBLIC_SUPABASE_URL);
const internalUrl = ensureHttps(process.env.SUPABASE_INTERNAL_URL);
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

console.log(
  '[Supabase Debug] Initializing client. isServer:', isServer,
  'URL:', supabaseUrl,
  'Anon Key (first 8):', supabaseAnonKey?.substring(0, 8)
);

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

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
