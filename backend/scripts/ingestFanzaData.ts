import axios from 'axios';
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.remote' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const fanzaApiId = process.env.FANZA_API_ID;
// API用とリンク用でアフィリエイトIDを分離（後方互換のためFANZA_AFFILIATE_IDも許容）
const fanzaApiAffiliateId = process.env.FANZA_API_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID; // 例: yotadata2-990
const fanzaLinkAffiliateId = process.env.FANZA_LINK_AFFILIATE_ID || process.env.FANZA_AFFILIATE_ID; // 例: yotadata2-001

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Supabase URL or Anon Key is not set.');
  process.exit(1);
}

if (!fanzaApiId || !fanzaApiAffiliateId) {
  console.error('FANZA_API_ID or FANZA_API_AFFILIATE_ID is not set in .env.remote');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseAnonKey);

function toFanzaAffiliate(rawUrl: string | null | undefined, affiliateId: string | undefined | null): string | null {
  if (!rawUrl) return null;
  if (rawUrl.startsWith('https://al.fanza.co.jp/')) return rawUrl;
  if (!affiliateId) return rawUrl;
  const lurl = encodeURIComponent(rawUrl);
  const aid = encodeURIComponent(affiliateId);
  return `https://al.fanza.co.jp/?lurl=${lurl}&af_id=${aid}&ch=link_tool&ch_id=link`;
}

async function fetchFanzaApiItems(queryParams: any) {
  const apiUrl = 'https://api.dmm.com/affiliate/v3/ItemList';
  const params = {
    api_id: fanzaApiId,
    affiliate_id: fanzaApiAffiliateId,
    site: 'FANZA',
    service: 'digital',
    floor: 'videoa',
    output: 'json',
    ...queryParams,
  };
  console.log('API Request Params:', JSON.stringify(params, null, 2));

  try {
    const response = await axios.get(apiUrl, { params });
    console.log('FANZA API Raw Response:', JSON.stringify(response.data, null, 2));
    return response.data.result;
  } catch (error: any) {
    console.error(`Error fetching data from FANZA API:`, error.message);
    if (error.response) {
      console.error('API Response Error:', error.response.data);
    }
    return null;
  }
}

function coalesce<T>(...vals: (T | undefined | null)[]): T | null {
  for (const v of vals) if (v != null) return v as T;
  return null;
}

function pickName(value: any): string | null {
  if (!value) return null;
  if (Array.isArray(value)) {
    const names = value
      .map((v) => (typeof v === 'string' ? v : v?.name))
      .filter(Boolean);
    return names.length ? names.join(' / ') : null;
  }
  return typeof value === 'string' ? value : value?.name ?? null;
}

async function insertVideoData(data: any) {
  // external_idで重複チェック
  const { data: existingVideo, error: fetchError } = await supabase
    .from('videos')
    .select('id')
    .eq('external_id', data.external_id)
    .single();

  if (fetchError && fetchError.code !== 'PGRST116') { // PGRST116はNot Found
    console.error('Error checking existing video:', fetchError);
    return;
  }

  if (existingVideo) {
    console.log(`Video with External ID ${data.external_id} already exists. Skipping insertion.`);
    return;
  }

  const { data: video, error: videoError } = await supabase
    .from('videos')
    .insert([
      {
        external_id: data.external_id,
        title: data.title,
        description: data.description ?? null,
        thumbnail_url: data.thumbnail_url ?? null,
        preview_video_url: data.preview_video_url ?? null,
        sample_video_url: data.sample_video_url ?? null,
        product_released_at: data.product_released_at ?? null,
        duration_seconds: data.duration_seconds ?? null,
        director: data.director ?? null,
        series: data.series ?? null,
        maker: data.maker ?? null,
        label: data.label ?? null,
        price: data.price ?? null,
        image_urls: data.image_urls ?? null,
        distribution_code: data.distribution_code ?? null,
        maker_code: data.maker_code ?? null,
        source: 'FANZA',
        product_url: data.product_url ?? null,
        published_at: data.published_at ?? null,
      },
    ])
    .select();

  if (videoError) {
    console.error('Error inserting video:', videoError);
    return;
  }

  const videoId = video[0].id;

  // タグの挿入と関連付け
  for (const genreName of data.genres ?? []) {
    const { data: tagRows, error: tagUpsertError } = await supabase
      .from('tags')
      .upsert([{ name: genreName }], { onConflict: 'name' })
      .select('id')
      .limit(1);
    if (tagUpsertError || !tagRows?.length) {
      console.error('Error upserting tag:', tagUpsertError);
      continue;
    }
    const tag = tagRows[0];
    const { error: videoTagError } = await supabase
      .from('video_tags')
      .insert([{ video_id: videoId, tag_id: tag.id }]);
    if (videoTagError) {
      console.error('Error inserting video_tag:', videoTagError);
    }
  }

  // 女優の挿入と関連付け
  for (const actress of data.actresses ?? []) {
    const name: string = typeof actress === 'string' ? actress : actress?.name;
    const fanzaId: string | null = typeof actress === 'object' ? actress?.id ?? null : null;

    let performerId: string | null = null;
    if (fanzaId) {
      const { data: pById } = await supabase
        .from('performers')
        .select('id')
        .eq('fanza_actress_id', fanzaId)
        .maybeSingle();
      performerId = pById?.id ?? null;
    }

    if (!performerId && name) {
      const { data: pByName } = await supabase
        .from('performers')
        .select('id')
        .eq('name', name)
        .maybeSingle();
      performerId = pByName?.id ?? null;
    }

    if (!performerId) {
      const { data: newPerformer, error: newPerformerError } = await supabase
        .from('performers')
        .upsert([
          { name, fanza_actress_id: fanzaId ?? undefined },
        ], { onConflict: 'fanza_actress_id' })
        .select('id')
        .limit(1);
      if (newPerformerError || !newPerformer?.length) {
        console.error('Error upserting performer:', newPerformerError);
        continue;
      }
      performerId = newPerformer[0].id;
    }

    if (performerId) {
      const { error: videoPerformerError } = await supabase
        .from('video_performers')
        .insert([{ video_id: videoId, performer_id: performerId }]);
      if (videoPerformerError) {
        console.error('Error inserting video_performer:', videoPerformerError);
      }
    }
  }

  console.log(`Video "${data.title}" (External ID: ${data.external_id}) and its associated data inserted successfully.`);
}

async function main() {
  const today = new Date();
  const oneYearAgo = new Date();
  oneYearAgo.setFullYear(today.getFullYear() - 1);

  const formatDate = (date: Date) => date.toISOString().split('T')[0];

  const gteReleaseDate = formatDate(oneYearAgo);
  const lteReleaseDate = formatDate(today);

  console.log(`Fetching videos released between ${gteReleaseDate} and ${lteReleaseDate}...`);

  let offset = 1;
  const hits = 100; // 最大取得件数
  let totalCount = 0;
  let fetchedCount = 0;

  while (true) {
    const result = await fetchFanzaApiItems({
      hits: hits,
      offset: offset,
      sort: 'date', // 発売日順
      gte_release_date: gteReleaseDate,
      lte_release_date: lteReleaseDate,
    });

    if (!result || !result.items || result.items.length === 0) {
      console.log('No more items to fetch or an error occurred.');
      break;
    }

    if (totalCount === 0) {
      totalCount = result.total_count;
      console.log(`Total videos found in the last year: ${totalCount}`);
    }

    for (const item of result.items) {
      // 収録時間のパース (例: "120分" -> 7200 秒)
      const volumeText: string | null = item.iteminfo?.volume ?? null;
      const durationMinutesMatch = typeof volumeText === 'string' ? volumeText.match(/(\d+)\s*分/) : null;
      const durationSeconds: number | null = durationMinutesMatch ? parseInt(durationMinutesMatch[1], 10) * 60 : null;

      // 価格のパース
      let parsedPrice: number | null = null;
      if (item.prices && item.prices.price != null) {
        const priceString = String(item.prices.price).replace(/[^0-9.]/g, '');
        const n = parseFloat(priceString);
        parsedPrice = isNaN(n) ? null : n;
      }
      console.log(`Item CID: ${item.content_id}, Raw Price: ${item.prices?.price}, Parsed Price: ${parsedPrice}`);

      // プレビュー動画URLの選択（利用可能なサイズの中から優先的に）
      const sm = item.sampleMovieURL ?? {};
      const previewUrl: string | null = coalesce(
        sm.size_720_480,
        sm.size_644_414,
        sm.size_560_360,
        sm.size_476_306
      );

      // サンプル画像URLリスト
      const imageUrls: string[] | null = coalesce(
        item.sampleImageURL?.sample_s?.image,
        item.sampleImageURL?.sample_l?.image
      );

      // ジャンルと女優の抽出
      const genres = item.iteminfo?.genre
        ? (Array.isArray(item.iteminfo.genre)
            ? item.iteminfo.genre.map((g: any) => g.name).filter(Boolean)
            : [item.iteminfo.genre.name])
        : [];
      const actressesRaw = item.iteminfo?.actress
        ? (Array.isArray(item.iteminfo.actress) ? item.iteminfo.actress : [item.iteminfo.actress])
        : [];

      const director = pickName(item.iteminfo?.director);
      const series = pickName(item.iteminfo?.series);
      const maker = pickName(item.iteminfo?.maker);
      const label = pickName(item.iteminfo?.label);

      const makerCode: string | null = (item as any).maker_product ?? item.product_id ?? null;
      const distributionCode: string | null = (item as any).jancode ?? null;

      const videoData = {
        external_id: item.content_id,
        title: item.title,
        description: null, // DMMのレスポンスには詳細説明がないため
        thumbnail_url: coalesce(item.imageURL?.large, item.imageURL?.list, item.imageURL?.small),
        preview_video_url: previewUrl,
        sample_video_url: previewUrl, // 同一のプレビューURLを格納
        product_released_at: item.date ?? null,
        duration_seconds: durationSeconds,
        director,
        series,
        maker,
        label,
        genres,
        actresses: actressesRaw, // 後段でIDも考慮して処理
        // 表示用リンクはリンク用アフィリエイトIDでラップ
        product_url: toFanzaAffiliate(item.URL, fanzaLinkAffiliateId),
        price: parsedPrice,
        image_urls: imageUrls,
        published_at: null,
        distribution_code: distributionCode,
        maker_code: makerCode,
      };

      await insertVideoData(videoData);
      fetchedCount++;
    }

    console.log(`Fetched ${fetchedCount} of ${totalCount} videos...`);

    // if (offset * hits >= totalCount) {
    //   console.log('All available items fetched.');
    //   break;
    // }

    offset += hits;
  }

  console.log('FANZA video ingestion completed.');
}

main();
