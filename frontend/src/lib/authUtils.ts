'use client';

// Force-clear Supabase auth tokens from localStorage as a last resort.
export function forceClearSupabaseAuth() {
  try {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
    const match = url.match(/^https?:\/\/(.*?)\.supabase\.co/);
    const projectRef = match ? match[1] : undefined;
    const keys = Object.keys(window.localStorage);
    for (const k of keys) {
      if (k.startsWith('sb-')) {
        // Supabase v2 default key prefix
        if (!projectRef || k.includes(projectRef)) {
          try { window.localStorage.removeItem(k); } catch {}
        }
      }
      if (k.includes('supabase.auth.token')) {
        // legacy key
        try { window.localStorage.removeItem(k); } catch {}
      }
    }
  } catch {}
}

