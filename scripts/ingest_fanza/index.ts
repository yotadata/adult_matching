import axios from 'axios';
import { createClient } from '@supabase/supabase-js';
import ws from 'ws';
import * as dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

const defaultEnvPath = 'docker/env/dev.env';
const envFilePath = process.env.INGEST_FANZA_ENV_FILE && process.env.INGEST_FANZA_ENV_FILE.trim().length > 0
  ? process.env.INGEST_FANZA_ENV_FILE
  : defaultEnvPath;
dotenv.config({ path: envFilePath });
console.log(`[ingest_fanza] Loaded environment from ${envFilePath}`);

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const fanzaApiId = process.env.FANZA_API_ID;
// API用とリンク用でアフィリエイトIDを分離（後方互換のためFANZA_AFFILIATE_IDも許容）
const fanzaApiAffiliateId = process.env.FANZA_API_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;
const fanzaLinkAffiliateId = process.env.FANZA_LINK_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Supabase URL or Anon Key is not set.');
  process.exit(1);
}

if (!fanzaApiId || !fanzaApiAffiliateId) {
  console.error('FANZA_API_ID or FANZA_API_AFFILIATE_ID is not set. Set it in docker/env/dev.env');
  process.exit(1);
}

const supabaseKey = supabaseServiceRoleKey || supabaseAnonKey;
if (!supabaseKey) {
  console.error('No Supabase key found. Set SUPABASE_SERVICE_ROLE_KEY or NEXT_PUBLIC_SUPABASE_ANON_KEY.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: { persistSession: false },
  realtime: { transport: ws },
});
if (!supabaseServiceRoleKey) {
  console.warn('[ingest_fanza] SUPABASE_SERVICE_ROLE_KEY not set. Falling back to anon key; RLS-protected tables may reject inserts.');
}
const debugLogs = process.env.INGEST_FANZA_DEBUG === '1';

type RawActress = string | { id?: string | null; name?: string | null };

type VideoInsertPayload = {
  external_id: string;
  title: string;
  description: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  preview_video_url: string | null;
  sample_video_url: string | null;
  product_released_at: string | null;
  duration_seconds: number | null;
  director: string | null;
  series: string | null;
  maker: string | null;
  label: string | null;
  product_url: string | null;
  affiliate_url: string | null;
  price: number | null;
  image_urls: string[] | null;
  distribution_code: string | null;
  maker_code: string | null;
  source: string;
};

type PreparedVideoData = VideoInsertPayload & {
  genres: string[];
  actresses: RawActress[];
};

const tagCache = new Map<string, string>();
const tagGroupIdCache = new Map<string, string>();
const performerNameCache = new Map<string, string>();
const performerFanzaCache = new Map<string, string>();
const DEFAULT_TAG_GROUP_NAME = '未分類';
const csvTagGroupMap = new Map<string, string>();

(() => {
  try {
    const csvPath = path.join(process.cwd(), 'docs', 'dmm_genres.csv');
    const raw = fs.readFileSync(csvPath, 'utf-8').trim();
    const lines = raw.split(/\r?\n/);
    for (const line of lines.slice(1)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const withoutQuotes = trimmed.replace(/^"|"$/g, '');
      const cells = withoutQuotes.split('","');
      if (cells.length < 2) continue;
      const [groupName, tagName] = cells;
      if (tagName) csvTagGroupMap.set(tagName, groupName);
    }
    console.log(`[ingest_fanza] Loaded ${csvTagGroupMap.size} tag->group mappings from docs/dmm_genres.csv`);
  } catch (error) {
    console.warn('[ingest_fanza] Failed to load docs/dmm_genres.csv. Missing file or unreadable?', error);
  }
})();

function toFanzaAffiliate(rawUrl: string | null | undefined, affiliateId: string | undefined | null): string | null {
  if (!rawUrl) return null;
  if (rawUrl.startsWith('https://al.fanza.co.jp/')) return rawUrl;
  if (!affiliateId) return rawUrl;
  const lurl = encodeURIComponent(rawUrl);
  const aid = encodeURIComponent(affiliateId);
  return `https://al.fanza.co.jp/?lurl=${lurl}&af_id=${aid}&ch=link_tool&ch_id=link`;
}

async function fetchFanzaApiItems(queryParams: Record<string, any>, service = 'digital', floor = 'videoa') {
  const apiUrl = 'https://api.dmm.com/affiliate/v3/ItemList';
  const params = {
    api_id: fanzaApiId,
    affiliate_id: fanzaApiAffiliateId,
    site: 'FANZA',
    service,
    floor,
    output: 'json',
    ...queryParams,
  };
  console.log(`[ingest_fanza] API params service=${service} floor=${floor} hits=${params.hits} offset=${params.offset} sort=${params.sort} gte=${params.gte_date ?? '-'} lte=${params.lte_date ?? '-'}`);

  try {
    const response = await axios.get(apiUrl, { params });
    return response.data.result;
  } catch (error: any) {
    console.error(`Error fetching data from FANZA API:`, error.message);
    if (error.response) {
      console.error('API Response Error:', JSON.stringify(error.response.data, null, 2));
    }
    return null;
  }
}

function coalesce<T>(...vals: (T | undefined | null)[]): T | null {
  for (const v of vals) if (v != null) return v as T;
  return null;
}

function normalizeFanzaDateTime(input: string | undefined | null): string | null {
  if (!input) return null;
  const raw = input.trim();
  if (!raw) return null;
  if (/^\d{8}$/.test(raw)) {
    return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}T00:00:00`;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return `${raw}T00:00:00`;
  if (/^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(:\d{2})?$/.test(raw)) {
    const [datePart, timePart] = raw.replace(/\s+/, ' ').split(' ');
    const time = timePart.length === 5 ? `${timePart}:00` : timePart;
    return `${datePart}T${time}`;
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(raw)) {
    return raw.length === 16 ? `${raw}:00` : raw;
  }
  return raw;
}

function pickName(value: any): string | null {
  if (!value) return null;
  if (Array.isArray(value)) {
    const names = value.map((v) => (typeof v === 'string' ? v : v?.name)).filter(Boolean);
    return names.length ? names.join(' / ') : null;
  }
  return typeof value === 'string' ? value : value?.name ?? null;
}

function unique<T>(array: (T | null | undefined)[]): T[] {
  return Array.from(new Set(array.filter((item): item is T => Boolean(item))));
}

function normalizeApiDate(input: string | undefined, boundary: 'start' | 'end'): string | undefined {
  if (!input) return undefined;
  const cleaned = input.trim();
  if (!cleaned) return undefined;
  let year: number, month: number, day: number;
  if (/^\d{8}$/.test(cleaned)) {
    year = Number(cleaned.slice(0, 4));
    month = Number(cleaned.slice(4, 6));
    day = Number(cleaned.slice(6, 8));
  } else if (/^\d{4}-\d{2}-\d{2}$/.test(cleaned)) {
    year = Number(cleaned.slice(0, 4));
    month = Number(cleaned.slice(5, 7));
    day = Number(cleaned.slice(8, 10));
  } else {
    console.warn(`[ingest_fanza] Unsupported date format "${input}".`);
    return cleaned;
  }
  const hh = boundary === 'start' ? '00' : '23';
  const mm = boundary === 'start' ? '00' : '59';
  const ss = boundary === 'start' ? '00' : '59';
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T${hh}:${mm}:${ss}`;
}

async function ensureTagId(name: string): Promise<string | null> {
  if (!name) return null;
  if (tagCache.has(name)) return tagCache.get(name) ?? null;

  const { data, error } = await supabase
    .from('tags')
    .upsert([{ name }], { onConflict: 'name' })
    .select('id, tag_group_id, name')
    .single();

  if (error || !data) {
    console.error('Error upserting tag:', error);
    return null;
  }

  tagCache.set(name, data.id);
  await ensureTagGroupAssignment(data.id, data.name ?? name, data.tag_group_id);
  return data.id;
}

async function ensureTagGroupAssignment(tagId: string, tagName: string, currentGroupId: string | null) {
  if (currentGroupId) return;
  const preferredGroupName = csvTagGroupMap.get(tagName) ?? DEFAULT_TAG_GROUP_NAME;
  const targetGroupId = await getTagGroupIdByName(preferredGroupName);
  if (!targetGroupId) return;

  const { error } = await supabase
    .from('tags')
    .update({ tag_group_id: targetGroupId })
    .eq('id', tagId)
    .is('tag_group_id', null);

  if (error) console.error(`[ingest_fanza] failed to update tag_group for tag_id=${tagId}:`, error);
}

async function getTagGroupIdByName(name: string): Promise<string | null> {
  if (tagGroupIdCache.has(name)) return tagGroupIdCache.get(name) ?? null;

  const { data, error } = await supabase
    .from('tag_groups')
    .select('id')
    .eq('name', name)
    .maybeSingle();

  if (error) {
    console.error(`[ingest_fanza] failed to fetch tag_group "${name}":`, error);
    tagGroupIdCache.set(name, '');
    return null;
  }

  if (!data) {
    tagGroupIdCache.set(name, '');
    return null;
  }

  tagGroupIdCache.set(name, data.id);
  return data.id;
}

async function ensurePerformerId(name: string | null, fanzaId: string | null): Promise<string | null> {
  const normalizedName = name?.trim() ?? null;

  if (fanzaId && performerFanzaCache.has(fanzaId)) return performerFanzaCache.get(fanzaId) ?? null;
  if (!fanzaId && normalizedName && performerNameCache.has(normalizedName)) return performerNameCache.get(normalizedName) ?? null;

  if (fanzaId) {
    const { data: existingByFanza, error: fanzaLookupError } = await supabase
      .from('performers')
      .select('id, name, fanza_actress_id')
      .eq('fanza_actress_id', fanzaId)
      .maybeSingle();
    if (fanzaLookupError) console.error('Error fetching performer by fanza_actress_id:', fanzaLookupError);
    if (existingByFanza) {
      const performerId = existingByFanza.id;
      if (existingByFanza.fanza_actress_id) performerFanzaCache.set(existingByFanza.fanza_actress_id, performerId);
      const n = existingByFanza.name?.trim();
      if (n) performerNameCache.set(n, performerId);
      return performerId;
    }
  }

  if (!fanzaId && normalizedName) {
    const { data: existingByName, error: nameLookupError } = await supabase
      .from('performers')
      .select('id, name, fanza_actress_id')
      .eq('name', normalizedName)
      .maybeSingle();
    if (nameLookupError) console.error('Error fetching performer by name:', nameLookupError);
    if (existingByName) {
      const performerId = existingByName.id;
      performerNameCache.set(normalizedName, performerId);
      const f = existingByName.fanza_actress_id?.trim();
      if (f) performerFanzaCache.set(f, performerId);
      return performerId;
    }
  }

  if (!normalizedName && !fanzaId) return null;

  const upsertPayload = fanzaId
    ? { name: normalizedName ?? fanzaId, fanza_actress_id: fanzaId }
    : { name: normalizedName };

  const { data: upserted, error: upsertError } = await supabase
    .from('performers')
    .upsert([upsertPayload], { onConflict: fanzaId ? 'fanza_actress_id' : 'name' })
    .select('id, name, fanza_actress_id')
    .single();

  if (upsertError || !upserted) {
    console.error('Error upserting performer:', upsertError);
    return null;
  }

  const performerId = upserted.id;
  const returnedName = upserted.name?.trim() ?? normalizedName;
  if (returnedName) performerNameCache.set(returnedName, performerId);
  const returnedFanza = upserted.fanza_actress_id ?? fanzaId;
  if (returnedFanza) performerFanzaCache.set(returnedFanza, performerId);
  if (normalizedName) performerNameCache.set(normalizedName, performerId);
  if (fanzaId) performerFanzaCache.set(fanzaId, performerId);

  return performerId;
}

function isDuplicateError(error: any): boolean {
  if (!error) return false;
  if (error.code === '23505') return true;
  const msg = typeof error.message === 'string' ? error.message : '';
  const detail = typeof error.details === 'string' ? error.details : '';
  return msg.includes('duplicate key') || detail.includes('duplicate key');
}

async function insertVideoData(data: PreparedVideoData): Promise<boolean> {
  const videoPayload: VideoInsertPayload = {
    external_id: data.external_id,
    title: data.title,
    description: data.description ?? null,
    thumbnail_url: data.thumbnail_url ?? null,
    thumbnail_vertical_url: data.thumbnail_vertical_url ?? null,
    preview_video_url: data.preview_video_url ?? null,
    sample_video_url: data.sample_video_url ?? null,
    product_released_at: data.product_released_at ?? null,
    duration_seconds: data.duration_seconds ?? null,
    director: data.director ?? null,
    series: data.series ?? null,
    maker: data.maker ?? null,
    label: data.label ?? null,
    product_url: data.product_url ?? null,
    affiliate_url: data.affiliate_url ?? null,
    price: data.price ?? null,
    image_urls: data.image_urls ?? null,
    distribution_code: data.distribution_code ?? null,
    maker_code: data.maker_code ?? null,
    source: data.source,
  };

  const { data: upsertedVideo, error: videoError } = await supabase
    .from('videos')
    .upsert([videoPayload], { onConflict: 'external_id' })
    .select('id')
    .single();

  if (videoError || !upsertedVideo) {
    console.error(`[ingest_fanza] video upsert failed external_id=${data.external_id}:`, videoError);
    return false;
  }

  const videoId = upsertedVideo.id;

  const uniqueGenres = unique<string>(data.genres ?? []);
  const tagRows: { video_id: string; tag_id: string }[] = [];
  for (const genreName of uniqueGenres) {
    const tagId = await ensureTagId(genreName);
    if (!tagId) continue;
    tagRows.push({ video_id: videoId, tag_id: tagId });
  }

  if (tagRows.length) {
    const { error: videoTagError } = await supabase
      .from('video_tags')
      .upsert(tagRows, { onConflict: 'video_id,tag_id' });
    if (videoTagError && !isDuplicateError(videoTagError)) {
      console.error(`[ingest_fanza] video_tags upsert failed video_id=${videoId}:`, videoTagError);
    }
  }

  const performerRows: { video_id: string; performer_id: string }[] = [];
  for (const actress of data.actresses ?? []) {
    const name = typeof actress === 'string' ? actress : actress?.name ?? null;
    const fanzaId = typeof actress === 'object' ? actress?.id ?? null : null;
    const performerId = await ensurePerformerId(name, fanzaId);
    if (!performerId) continue;
    performerRows.push({ video_id: videoId, performer_id: performerId });
  }

  if (performerRows.length) {
    const distinctRows = new Map<string, { video_id: string; performer_id: string }>();
    for (const row of performerRows) {
      const key = `${row.video_id}:${row.performer_id}`;
      if (!distinctRows.has(key)) distinctRows.set(key, row);
    }
    const { error: videoPerformerError } = await supabase
      .from('video_performers')
      .upsert(Array.from(distinctRows.values()), { onConflict: 'video_id,performer_id' });
    if (videoPerformerError && !isDuplicateError(videoPerformerError)) {
      console.error(`[ingest_fanza] video_performers upsert failed video_id=${videoId}:`, videoPerformerError);
    }
  }

  console.log(`[ingest_fanza] upserted external_id=${data.external_id} title="${data.title}"`);
  return true;
}

async function ingestSource(
  service: string,
  floor: string,
  source: string,
  gteReleaseDate: string,
  lteReleaseDate: string,
): Promise<{ success: number; failure: number; fetched: number }> {
  console.log(`\n[ingest_fanza] === Starting source: ${source} (${service}/${floor}) ===`);
  let offset = 1;
  const hits = 100;
  let totalCount = 0;
  let fetchedCount = 0;
  let successCount = 0;
  let failureCount = 0;

  while (true) {
    const apiGteDate = normalizeApiDate(gteReleaseDate, 'start');
    const apiLteDate = normalizeApiDate(lteReleaseDate, 'end');
    const query: Record<string, any> = { hits, offset, sort: '-date' };
    if (apiGteDate) query.gte_date = apiGteDate;
    if (apiLteDate) query.lte_date = apiLteDate;

    const result = await fetchFanzaApiItems(query, service, floor);

    if (!result || !result.items || result.items.length === 0) {
      console.log('[ingest_fanza] No more items to fetch or an error occurred.');
      break;
    }

    if (totalCount === 0) {
      totalCount = result.total_count;
      console.log(`[ingest_fanza] Total items found: ${totalCount}`);
    }

    for (const item of result.items) {
      const volumeText: string | null = item.iteminfo?.volume ?? null;
      const durationMinutesMatch = typeof volumeText === 'string' ? volumeText.match(/(\d+)\s*分/) : null;
      const durationSeconds: number | null = durationMinutesMatch ? parseInt(durationMinutesMatch[1], 10) * 60 : null;

      let parsedPrice: number | null = null;
      if (item.prices && item.prices.price != null) {
        const priceString = String(item.prices.price).replace(/[^0-9.]/g, '');
        const n = parseFloat(priceString);
        parsedPrice = isNaN(n) ? null : n;
        if (debugLogs) {
          console.log(`[ingest_fanza] price parsed external_id=${item.content_id} raw="${item.prices.price}" parsed=${parsedPrice}`);
        }
      }

      const sm = item.sampleMovieURL ?? {};
      const previewUrl: string | null = coalesce(
        sm.size_720_480,
        sm.size_644_414,
        sm.size_560_360,
        sm.size_476_306
      );

      const imageUrls: string[] | null = coalesce(
        item.sampleImageURL?.sample_s?.image,
        item.sampleImageURL?.sample_l?.image
      );
      const horizontalThumbnail: string | null = coalesce(
        item.imageURL?.large,
        item.imageURL?.list,
        item.imageURL?.small
      );
      const verticalThumbnail: string | null = coalesce(
        item.imageURL?.small,
        item.imageURL?.large,
        item.imageURL?.list
      );

      const genres = item.iteminfo?.genre
        ? (Array.isArray(item.iteminfo.genre)
            ? item.iteminfo.genre.map((g: any) => g.name).filter(Boolean)
            : [item.iteminfo.genre.name])
        : [];
      const actressesRaw: RawActress[] = item.iteminfo?.actress
        ? (Array.isArray(item.iteminfo.actress) ? item.iteminfo.actress : [item.iteminfo.actress])
        : [];

      const videoData: PreparedVideoData = {
        external_id: item.content_id,
        title: item.title,
        description: null,
        thumbnail_url: horizontalThumbnail,
        thumbnail_vertical_url: verticalThumbnail,
        preview_video_url: previewUrl,
        sample_video_url: previewUrl,
        product_released_at: normalizeFanzaDateTime(item.date) ?? null,
        duration_seconds: durationSeconds,
        director: pickName(item.iteminfo?.director),
        series: pickName(item.iteminfo?.series),
        maker: pickName(item.iteminfo?.maker),
        label: pickName(item.iteminfo?.label),
        genres,
        actresses: actressesRaw,
        product_url: item.URL ?? null,
        affiliate_url: item.affiliateURL ?? toFanzaAffiliate(item.URL, fanzaLinkAffiliateId),
        price: parsedPrice,
        image_urls: imageUrls,
        distribution_code: (item as any).jancode ?? null,
        maker_code: (item as any).maker_product ?? item.product_id ?? null,
        source,
      };

      const upserted = await insertVideoData(videoData);
      if (upserted) successCount++; else failureCount++;
      fetchedCount++;
    }

    console.log(`[ingest_fanza] fetched ${fetchedCount} / ${totalCount}`);

    if (offset + hits > 50000) {
      console.log('[ingest_fanza] Offset limit (50000) reached. Stopping pagination.');
      break;
    }
    offset += hits;
  }

  return { success: successCount, failure: failureCount, fetched: fetchedCount };
}

async function main() {
  const today = new Date();
  const formatDate = (date: Date) => date.toISOString().split('T')[0];

  let lteReleaseDate = formatDate(today);
  let gteReleaseDate: string;

  if (process.env.GTE_RELEASE_DATE) {
    gteReleaseDate = process.env.GTE_RELEASE_DATE;
    console.log(`Using custom gte_release_date: ${gteReleaseDate}`);
  } else {
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    gteReleaseDate = formatDate(oneYearAgo);
  }

  if (process.env.LTE_RELEASE_DATE) {
    lteReleaseDate = process.env.LTE_RELEASE_DATE;
    console.log(`Using custom lte_release_date: ${lteReleaseDate}`);
  }

  console.log(`Fetching videos released between ${gteReleaseDate} and ${lteReleaseDate}...`);

  const sources = [
    { service: 'digital', floor: 'videoa', source: 'FANZA' },
  ];

  let totalSuccess = 0;
  let totalFailure = 0;
  let totalFetched = 0;

  for (const { service, floor, source } of sources) {
    const result = await ingestSource(service, floor, source, gteReleaseDate, lteReleaseDate);
    totalSuccess += result.success;
    totalFailure += result.failure;
    totalFetched += result.fetched;
  }

  console.log(`[ingest_fanza] all sources completed success=${totalSuccess} failure=${totalFailure} totalFetched=${totalFetched}`);
}

main();
