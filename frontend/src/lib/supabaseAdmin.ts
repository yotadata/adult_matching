import { createClient } from '@supabase/supabase-js';

// Prefer internal URL on server to avoid localhost resolution issues inside containers
const isServer = typeof window === 'undefined';
const publicUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const internalUrl = process.env.SUPABASE_INTERNAL_URL;
const supabaseUrl = isServer ? (internalUrl || publicUrl) : publicUrl;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !serviceRoleKey) {
  throw new Error('Missing Supabase URL or Service Role Key. Ensure NEXT_PUBLIC_SUPABASE_URL (and SUPABASE_INTERNAL_URL for server) and SUPABASE_SERVICE_ROLE_KEY are set.');
}

export const supabaseAdmin = createClient(supabaseUrl, serviceRoleKey);
