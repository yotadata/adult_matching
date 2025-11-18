import axios from 'axios';
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

type BackfillOptions = {
  limit: number;
  chunkSize: number;
  dryRun: boolean;
};

const resolveEnvPath = (): string => {
  const overrides = [process.env.FANZA_BACKFILL_ENV_FILE, process.env.INGEST_FANZA_ENV_FILE];
  for (const candidate of overrides) {
    if (candidate && candidate.trim().length > 0) return candidate;
  }
  return 'docker/env/dev.env';
};

const envFilePath = resolveEnvPath();
const dotenvOverride = (process.env.DOTENV_CONFIG_OVERRIDE ?? '').toLowerCase() === 'true';
dotenv.config({ path: envFilePath, override: dotenvOverride });
const resolvedSupabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
if (!resolvedSupabaseUrl) {
  throw new Error('[fanza_backfill] Supabase base URL not set.');
}
if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
  process.env.NEXT_PUBLIC_SUPABASE_URL = resolvedSupabaseUrl;
}
if (!process.env.SUPABASE_URL) {
  process.env.SUPABASE_URL = resolvedSupabaseUrl;
}
const resolvedDbUrl = process.env.SUPABASE_DB_URL || process.env.REMOTE_DATABASE_URL;
if (resolvedDbUrl && !process.env.SUPABASE_DB_URL) {
  process.env.SUPABASE_DB_URL = resolvedDbUrl;
}
console.log(`[fanza_backfill] Loaded environment from ${envFilePath}`);

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;
const fanzaApiId = process.env.FANZA_API_ID;
const fanzaApiAffiliateId = process.env.FANZA_API_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;
const fanzaLinkAffiliateId = process.env.FANZA_LINK_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('[fanza_backfill] Missing Supabase URL or key. Set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.');
  process.exit(1);
}

if (!fanzaApiId || !fanzaApiAffiliateId) {
  console.error('[fanza_backfill] Missing FANZA API credentials. Set FANZA_API_ID and FANZA_API_AFFILIATE_ID.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: { persistSession: false },
});

const API_URL = 'https://api.dmm.com/affiliate/v3/ItemList';

const coalesce = <T>(...vals: Array<T | null | undefined>): T | null => {
  for (const val of vals) {
    if (val !== undefined && val !== null) return val as T;
  }
  return null;
};

const normalizeFanzaDateTime = (input: string | null | undefined): string | null => {
  if (!input) return null;
  const raw = input.trim();
  if (!raw) return null;
  if (/^\d{8}$/.test(raw)) {
    const year = raw.slice(0, 4);
    const month = raw.slice(4, 6);
    const day = raw.slice(6, 8);
    return `${year}-${month}-${day}T00:00:00`;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return `${raw}T00:00:00`;
  }
  if (/^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(:\d{2})?$/.test(raw)) {
    const [datePart, timePart] = raw.split(/\s+/);
    const time = timePart.length === 5 ? `${timePart}:00` : timePart;
    return `${datePart}T${time}`;
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(raw)) {
    return raw.length === 16 ? `${raw}:00` : raw;
  }
  return raw;
};

const toFanzaAffiliate = (rawUrl: string | null | undefined, affiliateId: string | undefined | null): string | null => {
  if (!rawUrl) return null;
  if (rawUrl.startsWith('https://al.fanza.co.jp/')) return rawUrl;
  if (!affiliateId) return rawUrl;
  const lurl = encodeURIComponent(rawUrl);
  const aid = encodeURIComponent(affiliateId);
  return `https://al.fanza.co.jp/?lurl=${lurl}&af_id=${aid}&ch=link_tool&ch_id=link`;
};

const parseArgs = (): BackfillOptions => {
  const args = process.argv.slice(2);
  const opts: BackfillOptions = { limit: 0, chunkSize: 10, dryRun: false };

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
};

const fetchTargetExternalIds = async (limit: number): Promise<string[]> => {
  const ids: string[] = [];
  const pageSize = 1000;
  const effectiveLimit = limit > 0 ? limit : Number.MAX_SAFE_INTEGER;
  let from = 0;

  while (from < effectiveLimit) {
    const to = Math.min(from + pageSize - 1, effectiveLimit - 1);
    const { data, error } = await supabase
      .from('videos')
      .select('external_id')
      .eq('source', 'FANZA')
      .or('thumbnail_vertical_url.is.null,affiliate_url.is.null')
      .order('created_at', { ascending: true })
      .range(from, to);

    if (error) {
      throw new Error(`[fanza_backfill] Failed to query target videos: ${error.message}`);
    }

    const batch = (data ?? [])
      .map((row) => row.external_id as string | null)
      .filter((id): id is string => Boolean(id));
    ids.push(...batch);
    if (!data || data.length < pageSize) break;
    from += pageSize;
  }

  return ids.slice(0, limit > 0 ? limit : ids.length);
};

const fetchItemsByCids = async (cids: string[]): Promise<any[]> => {
  if (cids.length === 0) return [];
  const params = {
    api_id: fanzaApiId,
    affiliate_id: fanzaApiAffiliateId,
    site: 'FANZA',
    service: 'digital',
    floor: 'videoa',
    output: 'json',
    hits: cids.length,
    cid: cids.join(','),
  };
  try {
    const response = await axios.get(API_URL, { params });
    const items = response.data?.result?.items;
    if (!Array.isArray(items)) return [];
    return items;
  } catch (error: any) {
    console.error('[fanza_backfill] FANZA API error:', error?.message ?? error);
    if (error?.response) {
      console.error(JSON.stringify(error.response.data, null, 2));
    }
    return [];
  }
};

const buildUpdatePayload = (item: any) => {
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
};

const main = async () => {
  const options = parseArgs();
  console.log(`[fanza_backfill] Target limit=${options.limit} chunkSize=${options.chunkSize} dryRun=${options.dryRun}`);

  const targets = await fetchTargetExternalIds(options.limit);
  if (targets.length === 0) {
    console.log('[fanza_backfill] No videos need backfill. Nothing to do.');
    return;
  }

  console.log(`[fanza_backfill] Found ${targets.length} candidate videos.`);
  const chunks: string[][] = [];
  for (let i = 0; i < targets.length; i += options.chunkSize) {
    chunks.push(targets.slice(i, i + options.chunkSize));
  }

  let updatedCount = 0;
  for (const chunk of chunks) {
    const items = await fetchItemsByCids(chunk);
    if (!items.length) {
      console.warn(`[fanza_backfill] No API items returned for chunk [${chunk.join(', ')}]`);
      continue;
    }
    const payloads = items
      .map(buildUpdatePayload)
      .filter((row) => typeof row.external_id === 'string' && row.external_id.length > 0);
    if (payloads.length === 0) continue;

    if (options.dryRun) {
      console.log('[fanza_backfill] Dry run payload:', payloads);
      updatedCount += payloads.length;
      continue;
    }

    const { error } = await supabase
      .from('videos')
      .upsert(payloads, { onConflict: 'external_id' });
    if (error) {
      console.error('[fanza_backfill] Upsert failed:', error.message);
      continue;
    }
    updatedCount += payloads.length;
    console.log(`[fanza_backfill] Updated ${payloads.length} videos (chunk).`);
  }

  console.log(`[fanza_backfill] Completed. Updated ${updatedCount} videos.`);
};

main().catch((error) => {
  console.error('[fanza_backfill] Unexpected error:', error);
  process.exit(1);
});
