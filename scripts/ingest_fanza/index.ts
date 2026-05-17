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

type BookInsertPayload = {
  external_id: string;
  title: string;
  description: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  sample_image_urls: string[] | null;
  page_count: number | null;
  author: string | null;
  author_id: string | null;
  series: string | null;
  publisher: string | null;
  label: string | null;
  price: number | null;
  product_url: string | null;
  affiliate_url: string | null;
  product_released_at: string | null;
  source: string;
};

type PreparedBookData = BookInsertPayload & {
  genres: string[];
};

const tagCache = new Map<string, string>();
const tagGroupIdCache = new Map<string, string>();
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

async function fetchFanzaApiItems(queryParams: Record<string, any>) {
  const apiUrl = 'https://api.dmm.com/affiliate/v3/ItemList';
  const params = {
    api_id: fanzaApiId,
    affiliate_id: fanzaApiAffiliateId,
    site: 'FANZA',
    service: 'ebook',   // 電子書籍
    floor: 'comic',     // 漫画
    output: 'json',
    ...queryParams,
  };
  console.log(`[ingest_fanza] API params hits=${params.hits} offset=${params.offset} sort=${params.sort} gte=${params.gte_date ?? '-'} lte=${params.lte_date ?? '-'}`);

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

function isDuplicateError(error: any): boolean {
  if (!error) return false;
  if (error.code === '23505') return true;
  const msg = typeof error.message === 'string' ? error.message : '';
  const detail = typeof error.details === 'string' ? error.details : '';
  return msg.includes('duplicate key') || detail.includes('duplicate key');
}

async function insertBookData(data: PreparedBookData): Promise<boolean> {
  const bookPayload: BookInsertPayload = {
    external_id: data.external_id,
    title: data.title,
    description: data.description ?? null,
    thumbnail_url: data.thumbnail_url ?? null,
    thumbnail_vertical_url: data.thumbnail_vertical_url ?? null,
    sample_image_urls: data.sample_image_urls ?? null,
    page_count: data.page_count ?? null,
    author: data.author ?? null,
    author_id: data.author_id ?? null,
    series: data.series ?? null,
    publisher: data.publisher ?? null,
    label: data.label ?? null,
    price: data.price ?? null,
    product_url: data.product_url ?? null,
    affiliate_url: data.affiliate_url ?? null,
    product_released_at: data.product_released_at ?? null,
    source: 'FANZA',
  };

  const { data: upsertedBook, error: bookError } = await supabase
    .from('books')
    .upsert([bookPayload], { onConflict: 'external_id' })
    .select('id')
    .single();

  if (bookError || !upsertedBook) {
    console.error(`[ingest_fanza] book upsert failed external_id=${data.external_id}:`, bookError);
    return false;
  }

  const bookId = upsertedBook.id;

  const uniqueGenres = unique<string>(data.genres ?? []);
  const tagRows: { book_id: string; tag_id: string }[] = [];
  for (const genreName of uniqueGenres) {
    const tagId = await ensureTagId(genreName);
    if (!tagId) continue;
    tagRows.push({ book_id: bookId, tag_id: tagId });
  }

  if (tagRows.length) {
    const { error: bookTagError } = await supabase
      .from('book_tags')
      .upsert(tagRows, { onConflict: 'book_id,tag_id' });
    if (bookTagError && !isDuplicateError(bookTagError)) {
      console.error(`[ingest_fanza] book_tags upsert failed book_id=${bookId}:`, bookTagError);
    }
  }

  console.log(`[ingest_fanza] upserted external_id=${data.external_id} title="${data.title}"`);
  return true;
}

async function main() {
  const today = new Date();
  const formatDate = (date: Date) => date.toISOString().split('T')[0];

  let lteReleaseDate = formatDate(today);
  let gteReleaseDate: string | undefined;

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

  console.log(`Fetching books released between ${gteReleaseDate} and ${lteReleaseDate}...`);

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

    const result = await fetchFanzaApiItems(query);

    if (!result || !result.items || result.items.length === 0) {
      console.log('No more items to fetch or an error occurred.');
      break;
    }

    if (totalCount === 0) {
      totalCount = result.total_count;
      console.log(`Total books found: ${totalCount}`);
    }

    for (const item of result.items) {
      // ページ数のパース（漫画APIのvolumeはページ数を返す場合がある）
      const volumeText: string | null = item.iteminfo?.volume ?? null;
      const pageCountMatch = typeof volumeText === 'string' ? volumeText.match(/(\d+)/) : null;
      const pageCount: number | null = pageCountMatch ? parseInt(pageCountMatch[1], 10) : null;

      // 価格のパース
      let parsedPrice: number | null = null;
      if (item.prices?.price != null) {
        const n = parseFloat(String(item.prices.price).replace(/[^0-9.]/g, ''));
        parsedPrice = isNaN(n) ? null : n;
        if (debugLogs) {
          console.log(`[ingest_fanza] price parsed external_id=${item.content_id} raw="${item.prices.price}" parsed=${parsedPrice}`);
        }
      }

      // サムネイル
      const thumbnailUrl: string | null = coalesce(
        item.imageURL?.large,
        item.imageURL?.list,
        item.imageURL?.small
      );
      const thumbnailVerticalUrl: string | null = coalesce(
        item.imageURL?.small,
        item.imageURL?.large,
        item.imageURL?.list
      );

      // サンプル画像（漫画の試し読みページ）
      const sampleImageUrls: string[] | null = coalesce(
        item.sampleImageURL?.sample_s?.image,
        item.sampleImageURL?.sample_l?.image
      );

      // 著者情報（漫画APIではauthorまたはartistで返ってくる）
      const authorRaw = item.iteminfo?.author ?? item.iteminfo?.artist ?? null;
      const authorName: string | null = pickName(authorRaw);
      const authorId: string | null = Array.isArray(authorRaw)
        ? (authorRaw[0]?.id ?? null)
        : (authorRaw?.id ?? null);

      // ジャンル
      const genres = item.iteminfo?.genre
        ? (Array.isArray(item.iteminfo.genre)
            ? item.iteminfo.genre.map((g: any) => g.name).filter(Boolean)
            : [item.iteminfo.genre.name])
        : [];

      const series = pickName(item.iteminfo?.series);
      const publisher = pickName(item.iteminfo?.maker);
      const label = pickName(item.iteminfo?.label);

      const bookData: PreparedBookData = {
        external_id: item.content_id,
        title: item.title,
        description: item.description ?? null,
        thumbnail_url: thumbnailUrl,
        thumbnail_vertical_url: thumbnailVerticalUrl,
        sample_image_urls: sampleImageUrls,
        page_count: pageCount,
        author: authorName,
        author_id: authorId,
        series,
        publisher,
        label,
        price: parsedPrice,
        product_url: item.URL ?? null,
        affiliate_url: item.affiliateURL ?? toFanzaAffiliate(item.URL, fanzaLinkAffiliateId),
        product_released_at: normalizeFanzaDateTime(item.date) ?? null,
        genres,
        source: 'FANZA',
      };

      const upserted = await insertBookData(bookData);
      if (upserted) { successCount++; } else { failureCount++; }
      fetchedCount++;
    }

    console.log(`[ingest_fanza] fetched ${fetchedCount} / ${totalCount}`);

    if (offset + hits > 50000) {
      console.log('[ingest_fanza] Offset limit (50000) reached. Stopping pagination.');
      break;
    }
    offset += hits;
  }

  console.log(`[ingest_fanza] completed success=${successCount} failure=${failureCount} totalFetched=${fetchedCount}`);
}

main();
