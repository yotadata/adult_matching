/**
 * ingest_mgs — MGS動画スクレイピング取り込みスクリプト
 *
 * 動作概要:
 *   1. MGS動画の一覧ページ（新着順）を巡回して product_id を収集
 *   2. 各作品の詳細ページから metaデータをスクレイピング
 *   3. Supabase の videos / performers / tags テーブルへ upsert
 *
 * 環境変数（docker/env/dev.env に追記）:
 *   MGS_AFFILIATE_ID   - MGSアフィリエイトID（?aff=XXX）
 *   GTE_RELEASE_DATE   - 取得開始日 (YYYY-MM-DD)。省略時は30日前
 *   LTE_RELEASE_DATE   - 取得終了日 (YYYY-MM-DD)。省略時は今日
 *   PAGE_LIMIT         - 最大ページ数（テスト用）。省略時は無制限
 *   SKIP_EXISTING      - '1' のとき DB に存在する作品をスキップ（デフォルト: 1）
 *   INGEST_MGS_DEBUG   - '1' のとき詳細ログ出力
 *   REQUEST_DELAY_MS   - リクエスト間の待機 ms（デフォルト: 800）
 */
import axios from 'axios';
import * as cheerio from 'cheerio';
import { createClient } from '@supabase/supabase-js';
import ws from 'ws';
import * as dotenv from 'dotenv';

const defaultEnvPath = 'docker/env/dev.env';
const envFilePath =
  process.env.INGEST_MGS_ENV_FILE?.trim() || defaultEnvPath;
dotenv.config({ path: envFilePath });
console.log(`[ingest_mgs] Loaded environment from ${envFilePath}`);

// ── 環境変数 ──────────────────────────────────────────
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey =
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const mgsAffiliateId = process.env.MGS_AFFILIATE_ID ?? '';

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('[ingest_mgs] Supabase URL or Anon Key is not set.');
  process.exit(1);
}

const supabaseKey = supabaseServiceRoleKey || supabaseAnonKey;
const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: { persistSession: false },
  realtime: { transport: ws },
});
if (!supabaseServiceRoleKey) {
  console.warn(
    '[ingest_mgs] SUPABASE_SERVICE_ROLE_KEY not set. Falling back to anon key.',
  );
}

const debugLogs = process.env.INGEST_MGS_DEBUG === '1';
const skipExisting = process.env.SKIP_EXISTING !== '0'; // デフォルト: スキップする
const requestDelayMs = Number(process.env.REQUEST_DELAY_MS ?? '800');
const pageLimit = process.env.PAGE_LIMIT ? Number(process.env.PAGE_LIMIT) : Infinity;

// ── HTTP クライアント（年齢確認Cookie付き） ──────────────
const httpClient = axios.create({
  baseURL: 'https://www.mgstage.com',
  timeout: 15_000,
  headers: {
    'User-Agent':
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
      '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    Accept:
      'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    Cookie: 'mgs_agef=1', // 年齢確認突破
  },
});

// ── 型定義 ─────────────────────────────────────────────
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
  performers: string[];
};

// ── キャッシュ ─────────────────────────────────────────
const tagCache = new Map<string, string>();
const performerNameCache = new Map<string, string>();

// ── ユーティリティ ─────────────────────────────────────
function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function unique<T>(arr: (T | null | undefined)[]): T[] {
  return Array.from(new Set(arr.filter((v): v is T => Boolean(v))));
}

function isDuplicateError(error: any): boolean {
  if (!error) return false;
  if (error.code === '23505') return true;
  const msg = String(error.message ?? '');
  const detail = String(error.details ?? '');
  return msg.includes('duplicate key') || detail.includes('duplicate key');
}

/**
 * "62min" "120min" → 秒数
 * "1:02:00" → 秒数
 */
function parseDurationSeconds(raw: string | null): number | null {
  if (!raw) return null;
  const minMatch = raw.match(/(\d+)\s*min/i);
  if (minMatch) return parseInt(minMatch[1], 10) * 60;
  const hmsMatch = raw.match(/(\d+):(\d{2}):(\d{2})/);
  if (hmsMatch)
    return (
      parseInt(hmsMatch[1], 10) * 3600 +
      parseInt(hmsMatch[2], 10) * 60 +
      parseInt(hmsMatch[3], 10)
    );
  const msMatch = raw.match(/(\d+):(\d{2})/);
  if (msMatch)
    return parseInt(msMatch[1], 10) * 60 + parseInt(msMatch[2], 10);
  return null;
}

/**
 * "2026/06/11" → "2026-06-11T00:00:00"
 */
function parseMgsDate(raw: string | null): string | null {
  if (!raw) return null;
  const m = raw.trim().match(/(\d{4})\/(\d{2})\/(\d{2})/);
  if (!m) return null;
  return `${m[1]}-${m[2]}-${m[3]}T00:00:00`;
}

/**
 * 価格文字列 → 数値（円）
 * "476円～" → 476
 */
function parsePrice(raw: string | null): number | null {
  if (!raw) return null;
  const m = raw.match(/[\d,]+/);
  if (!m) return null;
  const n = parseInt(m[0].replace(/,/g, ''), 10);
  return isNaN(n) ? null : n;
}

/** MGSアフィリエイトURL生成 */
function toMgsAffiliateUrl(productId: string, affiliateId: string): string | null {
  if (!affiliateId) return null;
  return `https://www.mgstage.com/product/product_detail/${productId}/?aff=${encodeURIComponent(affiliateId)}`;
}

// ── Supabase helpers ───────────────────────────────────
async function ensureTagId(name: string): Promise<string | null> {
  if (!name.trim()) return null;
  if (tagCache.has(name)) return tagCache.get(name)!;

  const { data, error } = await supabase
    .from('tags')
    .upsert([{ name }], { onConflict: 'name' })
    .select('id')
    .single();

  if (error || !data) {
    console.error('[ingest_mgs] tag upsert failed:', error);
    return null;
  }
  tagCache.set(name, data.id);
  return data.id;
}

async function ensurePerformerId(name: string): Promise<string | null> {
  const normalized = name.trim();
  if (!normalized) return null;
  if (performerNameCache.has(normalized)) return performerNameCache.get(normalized)!;

  const { data: existing } = await supabase
    .from('performers')
    .select('id')
    .eq('name', normalized)
    .maybeSingle();

  if (existing) {
    performerNameCache.set(normalized, existing.id);
    return existing.id;
  }

  const { data: upserted, error } = await supabase
    .from('performers')
    .upsert([{ name: normalized }], { onConflict: 'name' })
    .select('id')
    .single();

  if (error || !upserted) {
    console.error('[ingest_mgs] performer upsert failed:', error);
    return null;
  }
  performerNameCache.set(normalized, upserted.id);
  return upserted.id;
}

async function videoExists(externalId: string): Promise<boolean> {
  const { data } = await supabase
    .from('videos')
    .select('id')
    .eq('external_id', externalId)
    .maybeSingle();
  return Boolean(data);
}

async function insertVideoData(data: PreparedVideoData): Promise<boolean> {
  const videoPayload: VideoInsertPayload = {
    external_id: data.external_id,
    title: data.title,
    description: data.description,
    thumbnail_url: data.thumbnail_url,
    thumbnail_vertical_url: data.thumbnail_vertical_url,
    preview_video_url: data.preview_video_url,
    sample_video_url: data.sample_video_url,
    product_released_at: data.product_released_at,
    duration_seconds: data.duration_seconds,
    director: data.director,
    series: data.series,
    maker: data.maker,
    label: data.label,
    product_url: data.product_url,
    affiliate_url: data.affiliate_url,
    price: data.price,
    image_urls: data.image_urls,
    distribution_code: data.distribution_code,
    maker_code: data.maker_code,
    source: data.source,
  };

  const { data: upserted, error } = await supabase
    .from('videos')
    .upsert([videoPayload], { onConflict: 'external_id' })
    .select('id')
    .single();

  if (error || !upserted) {
    console.error(`[ingest_mgs] video upsert failed external_id=${data.external_id}:`, error);
    return false;
  }

  const videoId = upserted.id;

  // タグ
  const tagRows: { video_id: string; tag_id: string }[] = [];
  for (const genre of unique<string>(data.genres)) {
    const tagId = await ensureTagId(genre);
    if (tagId) tagRows.push({ video_id: videoId, tag_id: tagId });
  }
  if (tagRows.length) {
    const { error: te } = await supabase
      .from('video_tags')
      .upsert(tagRows, { onConflict: 'video_id,tag_id' });
    if (te && !isDuplicateError(te))
      console.error(`[ingest_mgs] video_tags upsert failed:`, te);
  }

  // 出演者
  const perfRows: { video_id: string; performer_id: string }[] = [];
  for (const name of unique<string>(data.performers)) {
    const perfId = await ensurePerformerId(name);
    if (perfId) perfRows.push({ video_id: videoId, performer_id: perfId });
  }
  if (perfRows.length) {
    const { error: pe } = await supabase
      .from('video_performers')
      .upsert(perfRows, { onConflict: 'video_id,performer_id' });
    if (pe && !isDuplicateError(pe))
      console.error(`[ingest_mgs] video_performers upsert failed:`, pe);
  }

  if (debugLogs) {
    console.log(`[ingest_mgs] upserted ${data.external_id} "${data.title}"`);
  } else {
    process.stdout.write('.');
  }
  return true;
}

// ── スクレイピング ─────────────────────────────────────

/**
 * 一覧ページから product_id のリストを取得
 * URL: /search/cSearch.php?sort=new&list_cnt=120&type=top&page=N&sale_start_range=YYYY.MM.DD-YYYY.MM.DD
 */
async function fetchListPage(page: number, gteDate: string, lteDate: string): Promise<string[]> {
  // YYYY-MM-DD → YYYY.MM.DD
  const from = gteDate.replace(/-/g, '.');
  const to = lteDate.replace(/-/g, '.');

  const params = new URLSearchParams({
    sort: 'new',
    list_cnt: '120',
    type: 'top',
    page: String(page),
    sale_start_range: `${from}-${to}`,
  });

  const url = `/search/cSearch.php?${params.toString()}`;
  if (debugLogs) console.log(`[ingest_mgs] GET ${url}`);

  let html: string;
  try {
    const res = await httpClient.get(url);
    html = res.data as string;
  } catch (err: any) {
    console.error(`[ingest_mgs] fetch list page=${page} failed:`, err.message);
    return [];
  }

  const $ = cheerio.load(html);
  const productIds: string[] = [];

  $('ul.product_list li.product_list_item h5 a, div.rank_list li h5 a').each((_, el) => {
    const href = $(el).attr('href') ?? '';
    const m = href.match(/\/product\/product_detail\/([^/]+)\//);
    if (m) productIds.push(m[1]);
  });

  if (debugLogs) console.log(`[ingest_mgs] page=${page} found ${productIds.length} products`);
  return productIds;
}

/**
 * 一覧ページから総作品数を取得（1ページ目のみ）
 */
async function fetchTotalCount(gteDate: string, lteDate: string): Promise<number> {
  const from = gteDate.replace(/-/g, '.');
  const to = lteDate.replace(/-/g, '.');
  const params = new URLSearchParams({
    sort: 'new', list_cnt: '120', type: 'top', page: '1',
    sale_start_range: `${from}-${to}`,
  });

  try {
    const res = await httpClient.get(`/search/cSearch.php?${params.toString()}`);
    const $ = cheerio.load(res.data as string);
    // "全XXX件" または "XXX件" 形式
    const text = $('div.search_result_count, p.result_count, .total_count').first().text();
    const m = text.match(/([\d,]+)\s*件/);
    if (m) return parseInt(m[1].replace(/,/g, ''), 10);
    // フォールバック: リストが空なら0
    return 0;
  } catch {
    return 0;
  }
}

/**
 * 詳細ページから動画メタデータをスクレイピング
 */
async function fetchProductDetail(productId: string): Promise<PreparedVideoData | null> {
  const url = `/product/product_detail/${productId}/`;
  if (debugLogs) console.log(`[ingest_mgs] GET detail ${url}`);

  let html: string;
  try {
    const res = await httpClient.get(url);
    html = res.data as string;
  } catch (err: any) {
    console.error(`[ingest_mgs] fetch detail ${productId} failed:`, err.message);
    return null;
  }

  const $ = cheerio.load(html);

  // タイトル
  const title = $('h1.tag').first().text().trim() ||
    $('div.detail_data h1').first().text().trim() ||
    $('title').text().replace(/\s*\|\s*MGS動画.*$/, '').trim();

  if (!title) {
    console.warn(`[ingest_mgs] no title found for ${productId}`);
    return null;
  }

  // ジャケット画像（縦横共通 - MGSは正方形ジャケット）
  const jacketImgSrc =
    $('div.detail_data h2 img.enlarge_image').attr('src') ||
    $('div.detail_data h2 img').first().attr('src') ||
    null;

  // pb_p_（縮小）→ pb_e_（拡大）に変換
  const thumbnailLarge = jacketImgSrc
    ? jacketImgSrc.replace('pb_p_', 'pb_e_')
    : null;
  const thumbnailSmall = jacketImgSrc ?? null;

  // キャプチャ画像一覧（cap_t1_N_ 形式）
  const imageUrls: string[] = [];
  $('div.sample_img img, ul.sample_images img, div#cap_image img').each((_, el) => {
    const src = $(el).attr('src') || $(el).attr('data-src');
    if (src && src.includes('image.mgstage.com')) imageUrls.push(src);
  });

  // 詳細テーブルのパース
  const meta: Record<string, string> = {};
  $('div.detail_data table tr, div.info table tr').each((_, tr) => {
    const th = $(tr).find('th').first().text().trim();
    const td = $(tr).find('td').first().text().trim();
    if (th) meta[th] = td;
  });

  if (debugLogs) console.log(`[ingest_mgs] meta for ${productId}:`, meta);

  // 出演者
  const performerNames: string[] = [];
  $('div.detail_data table tr, div.info table tr').each((_, tr) => {
    const th = $(tr).find('th').first().text().trim();
    if (th === '出演') {
      $(tr).find('td a').each((_, a) => {
        const name = $(a).text().trim();
        if (name) performerNames.push(name);
      });
      // リンクなしの場合はテキスト全体を分割
      if (performerNames.length === 0) {
        const tdText = $(tr).find('td').first().text().trim();
        if (tdText) performerNames.push(...tdText.split(/[,、\s]+/).filter(Boolean));
      }
    }
  });

  // ジャンル
  const genres: string[] = [];
  $('div.detail_data table tr, div.info table tr').each((_, tr) => {
    const th = $(tr).find('th').first().text().trim();
    if (th === 'ジャンル') {
      $(tr).find('td a').each((_, a) => {
        const g = $(a).text().trim();
        if (g) genres.push(g);
      });
      if (genres.length === 0) {
        const tdText = $(tr).find('td').first().text().trim();
        if (tdText) genres.push(...tdText.split(/[\s　]+/).filter(Boolean));
      }
    }
  });

  // 価格（最安値を使用）
  const priceRaw = meta['価格'] || meta['販売価格'] || meta['ポイント'] || null;
  const price = parsePrice(priceRaw);

  // 配信開始日
  const releasedAt = parseMgsDate(meta['配信開始日'] || meta['発売日'] || null);

  // 収録時間
  const durationSeconds = parseDurationSeconds(meta['収録時間'] || null);

  // 品番（product_id と一致するはずだが念のため取得）
  const productCode = (meta['品番'] || productId).trim();

  const productUrl = `https://www.mgstage.com/product/product_detail/${productId}/`;
  const affiliateUrl = toMgsAffiliateUrl(productId, mgsAffiliateId);

  return {
    external_id: `mgs_${productId}`,
    title,
    description: null,
    thumbnail_url: thumbnailLarge ?? thumbnailSmall,
    thumbnail_vertical_url: thumbnailSmall ?? thumbnailLarge,
    preview_video_url: null,
    sample_video_url: null,
    product_released_at: releasedAt,
    duration_seconds: durationSeconds,
    director: meta['監督']?.trim() || null,
    series: meta['シリーズ']?.trim() || null,
    maker: meta['メーカー']?.trim() || null,
    label: meta['レーベル']?.trim() || null,
    product_url: productUrl,
    affiliate_url: affiliateUrl,
    price,
    image_urls: imageUrls.length > 0 ? imageUrls : null,
    distribution_code: null,
    maker_code: productCode,
    source: 'mgs',
    genres,
    performers: performerNames,
  };
}

// ── メイン ─────────────────────────────────────────────
async function main() {
  const today = new Date();
  const fmt = (d: Date) => d.toISOString().split('T')[0];

  const lteDate = process.env.LTE_RELEASE_DATE?.trim() || fmt(today);
  let gteDate: string;
  if (process.env.GTE_RELEASE_DATE?.trim()) {
    gteDate = process.env.GTE_RELEASE_DATE.trim();
  } else {
    const d = new Date();
    d.setDate(today.getDate() - 30);
    gteDate = fmt(d);
  }

  console.log(`[ingest_mgs] 取得期間: ${gteDate} 〜 ${lteDate}`);
  console.log(`[ingest_mgs] skipExisting=${skipExisting} pageLimit=${pageLimit} delay=${requestDelayMs}ms`);
  if (!mgsAffiliateId) {
    console.warn('[ingest_mgs] MGS_AFFILIATE_ID が未設定です。アフィリエイトURLなしで取り込みます。');
  }

  let page = 1;
  let totalSuccess = 0;
  let totalSkip = 0;
  let totalFailure = 0;
  let totalFetched = 0;

  while (page <= pageLimit) {
    const productIds = await fetchListPage(page, gteDate, lteDate);

    if (productIds.length === 0) {
      console.log(`\n[ingest_mgs] page=${page} 件数0のため終了`);
      break;
    }

    console.log(`\n[ingest_mgs] page=${page} / ${productIds.length}件処理開始`);

    for (const productId of productIds) {
      const externalId = `mgs_${productId}`;

      // 既存チェック
      if (skipExisting && await videoExists(externalId)) {
        if (debugLogs) console.log(`[ingest_mgs] skip existing: ${externalId}`);
        totalSkip++;
        continue;
      }

      await sleep(requestDelayMs);

      const detail = await fetchProductDetail(productId);
      if (!detail) {
        totalFailure++;
        continue;
      }

      const ok = await insertVideoData(detail);
      if (ok) totalSuccess++; else totalFailure++;
      totalFetched++;
    }

    page++;
  }

  console.log(
    `\n[ingest_mgs] 完了 success=${totalSuccess} skip=${totalSkip} failure=${totalFailure} fetched=${totalFetched}`,
  );
}

main().catch((err) => {
  console.error('[ingest_mgs] 予期しないエラー:', err);
  process.exit(1);
});
