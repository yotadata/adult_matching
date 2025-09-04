import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

// Load envs from .env.remote preferentially
import dotenv from 'dotenv';
dotenv.config({ path: '.env.remote' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;
const linkAffiliateId = process.env.FANZA_LINK_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Missing Supabase envs. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.remote');
  process.exit(1);
}

if (!linkAffiliateId) {
  console.error('Missing FANZA_LINK_AFFILIATE_ID (or fallback FANZA_AFFILIATE_ID) in .env.remote');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, serviceRoleKey || supabaseAnonKey);

function toFanzaAffiliate(rawUrl, afId) {
  if (!rawUrl) return rawUrl;
  if (rawUrl.startsWith('https://al.fanza.co.jp/')) return rawUrl;
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(rawUrl)}&af_id=${encodeURIComponent(afId)}&ch=link_tool&ch_id=link`;
}

async function backfill() {
  const pageSize = 500;
  let offset = 0;
  let updated = 0;
  let scanned = 0;

  console.log('Starting backfill of affiliate links...');

  while (true) {
    const { data: rows, error } = await supabase
      .from('videos')
      .select('id, product_url')
      .not('product_url', 'is', null)
      .not('product_url', 'ilike', 'https://al.fanza.co.jp/%')
      .range(offset, offset + pageSize - 1);

    if (error) {
      console.error('Fetch error:', error);
      break;
    }

    if (!rows || rows.length === 0) {
      break;
    }

    for (const row of rows) {
      scanned++;
      const newUrl = toFanzaAffiliate(row.product_url, linkAffiliateId);
      if (newUrl && newUrl !== row.product_url) {
        const { error: upErr } = await supabase
          .from('videos')
          .update({ product_url: newUrl })
          .eq('id', row.id);
        if (upErr) {
          console.error(`Update failed for id=${row.id}:`, upErr.message);
        } else {
          updated++;
        }
      }
    }

    console.log(`Scanned ${scanned} rows, updated ${updated} so far...`);
    offset += pageSize;
  }

  console.log(`Backfill completed. Total scanned: ${scanned}, updated: ${updated}.`);
}

backfill().catch((e) => {
  console.error('Unexpected error:', e);
  process.exit(1);
});
