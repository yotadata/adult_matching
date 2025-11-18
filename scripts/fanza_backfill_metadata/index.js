const { createClient } = require('@supabase/supabase-js');
const dotenv = require('dotenv');

const API_URL = 'https://api.dmm.com/affiliate/v3/ItemList';

function resolveEnvPath() {
  const overrides = [process.env.FANZA_BACKFILL_ENV_FILE, process.env.INGEST_FANZA_ENV_FILE];
  for (const file of overrides) {
    if (file && file.trim().length > 0) return file;
  }
  return 'docker/env/dev.env';
}

const envFilePath = resolveEnvPath();
const overrideEnv = (process.env.DOTENV_CONFIG_OVERRIDE ?? 'false').toLowerCase() === 'true';
dotenv.config({ path: envFilePath, override: overrideEnv });
console.log(`[fanza_backfill] Loaded environment from ${envFilePath}`);

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!supabaseUrl || !supabaseServiceKey) {
  console.error('[fanza_backfill] NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.');
  process.exit(1);
}

const fanzaApiId = process.env.FANZA_API_ID;
const fanzaApiAffiliateId = process.env.FANZA_API_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;
const fanzaLinkAffiliateId = process.env.FANZA_LINK_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;
if (!fanzaApiId || !fanzaApiAffiliateId) {
  console.error('[fanza_backfill] Missing FANZA API credentials.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: { persistSession: false },
});

const coalesce = (...values) => {
  for (const value of values) {
    if (value !== undefined && value !== null) return value;
  }
  return null;
};

function normalizeFanzaDateTime(input) {
  if (!input) return null;
  const raw = String(input).trim();
  if (!raw) return null;
  if (/^\d{8}$/.test(raw)) {
    const y = raw.slice(0, 4);
    const m = raw.slice(4, 6);
    const d = raw.slice(6, 8);
    return `${y}-${m}-${d}T00:00:00`;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return `${raw}T00:00:00`;
  if (/^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(:\d{2})?$/.test(raw)) {
    const [datePart, timePartRaw] = raw.split(/\s+/);
    const timePart = timePartRaw.length === 5 ? `${timePartRaw}:00` : timePartRaw;
    return `${datePart}T${timePart}`;
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(raw)) {
    return raw.length === 16 ? `${raw}:00` : raw;
  }
  return raw;
}

function toFanzaAffiliate(rawUrl, affiliateId) {
  if (!rawUrl) return null;
  if (rawUrl.startsWith('https://al.fanza.co.jp/')) return rawUrl;
  if (!affiliateId) return rawUrl;
  const params = new URLSearchParams({
    lurl: rawUrl,
    af_id: affiliateId,
    ch: 'link_tool',
    ch_id: 'link',
  });
  return `https://al.fanza.co.jp/?${params.toString()}`;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { limit: 0, chunkSize: 10, dryRun: false };
  while (args.length > 0) {
    const arg = args.shift();
    if (!arg) break;
    if (arg === '--limit' && args.length > 0) {
      const value = Number(args.shift());
      if (!Number.isNaN(value) && value > 0) opts.limit = Math.floor(value);
    } else if (arg === '--chunk-size' && args.length > 0) {
      const value = Number(args.shift());
      if (!Number.isNaN(value) && value > 0) opts.chunkSize = Math.max(1, Math.floor(value));
    } else if (arg === '--dry-run') {
      opts.dryRun = true;
    } else {
      console.warn(`[fanza_backfill] Unknown argument "${arg}" ignored.`);
    }
  }
  return opts;
}

async function fetchTargetExternalIds(limit) {
  const ids = [];
  const pageSize = 1000;
  const effectiveLimit = limit > 0 ? limit : Number.MAX_SAFE_INTEGER;
  let start = 0;

  while (start < effectiveLimit) {
    const end = start + Math.min(pageSize, effectiveLimit - start) - 1;
    const { data, error } = await supabase
      .from('videos')
      .select('external_id')
      .eq('source', 'FANZA')
      .or('thumbnail_vertical_url.is.null,affiliate_url.is.null')
      .order('created_at', { ascending: true })
      .range(start, end);

    if (error) {
      throw new Error(`[fanza_backfill] Failed to fetch candidate videos: ${error.message}`);
    }

    (data ?? []).forEach((row) => {
      if (row.external_id) ids.push(row.external_id);
    });

    if (!data || data.length < (end - start + 1)) break;
    start += data.length;
  }

  return ids;
}

async function fetchItemsByCids(cids) {
  const items = [];
  for (const cid of cids) {
    if (!cid) continue;
    const params = new URLSearchParams({
      api_id: fanzaApiId,
      affiliate_id: fanzaApiAffiliateId,
      site: 'FANZA',
      service: 'digital',
      floor: 'videoa',
      output: 'json',
      hits: '1',
      cid,
    });
    try {
      const response = await fetch(`${API_URL}?${params.toString()}`);
      if (!response.ok) {
        console.error(`[fanza_backfill] FANZA API error for cid ${cid}: HTTP ${response.status}`);
        continue;
      }
      const body = await response.json();
      const resultItems = body?.result?.items;
      if (Array.isArray(resultItems) && resultItems.length > 0) {
        items.push(resultItems[0]);
      }
    } catch (error) {
      console.error(`[fanza_backfill] FANZA API error for cid ${cid}:`, error?.message ?? error);
    }
  }
  return items;
}

function buildUpdatePayload(item) {
  const horizontalThumbnail = coalesce(item?.imageURL?.large, item?.imageURL?.list, item?.imageURL?.small);
  const verticalThumbnail = coalesce(item?.imageURL?.small, item?.imageURL?.large, item?.imageURL?.list);
  return {
    external_id: item?.content_id,
    thumbnail_url: horizontalThumbnail ?? null,
    thumbnail_vertical_url: verticalThumbnail ?? null,
    product_url: item?.URL ?? null,
    affiliate_url: item?.affiliateURL ?? toFanzaAffiliate(item?.URL, fanzaLinkAffiliateId),
    product_released_at: normalizeFanzaDateTime(item?.date),
  };
}

async function updateVideo(row) {
  const { error } = await supabase
    .from('videos')
    .update({
      thumbnail_url: row.thumbnail_url,
      thumbnail_vertical_url: row.thumbnail_vertical_url,
      product_url: row.product_url,
      affiliate_url: row.affiliate_url,
      product_released_at: row.product_released_at,
    })
    .eq('external_id', row.external_id);
  if (error) throw new Error(error.message);
}

async function main() {
  const options = parseArgs();
  console.log(`[fanza_backfill] Target limit=${options.limit} chunkSize=${options.chunkSize} dryRun=${options.dryRun}`);

  const targets = await fetchTargetExternalIds(options.limit);
  if (targets.length === 0) {
    console.log('[fanza_backfill] No videos need backfill. Nothing to do.');
    return;
  }

  console.log(`[fanza_backfill] Found ${targets.length} candidate videos.`);
  if (options.dryRun) {
    console.log(`[fanza_backfill] Dry run: ${targets.length} videos need backfill (API calls skipped).`);
    return;
  }

  const chunks = [];
  for (let i = 0; i < targets.length; i += options.chunkSize) {
    chunks.push(targets.slice(i, i + options.chunkSize));
  }

  let updatedCount = 0;
  for (const chunk of chunks) {
    const processed = Math.min(targets.length, updatedCount + chunk.length);
    const percent = Math.min(100, Math.round((processed / targets.length) * 100));

    const items = await fetchItemsByCids(chunk);
    if (!items.length) {
      console.warn(`[${percent}%] No API items returned for chunk [${chunk.join(', ')}]`);
      continue;
    }

    const payloads = items
      .map(buildUpdatePayload)
      .filter((row) => typeof row.external_id === 'string' && row.external_id.length > 0);
    if (payloads.length === 0) continue;

    let success = true;
    for (const row of payloads) {
      try {
        await updateVideo(row);
      } catch (error) {
        success = false;
        console.error(`[${percent}%] Update failed for ${row.external_id}:`, error.message);
      }
    }

    if (success) {
      updatedCount += payloads.length;
      console.log(`[${percent}%] Updated ${payloads.length} videos (chunk).`);
    }
  }

  console.log(`[fanza_backfill] Completed. Updated ${updatedCount} videos.`);
}

main().catch((error) => {
  console.error('[fanza_backfill] Unexpected error:', error);
  process.exit(1);
});
