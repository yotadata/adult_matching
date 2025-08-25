import axios from 'axios';
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const fanzaApiId = process.env.FANZA_API_ID;
const fanzaAffiliateId = process.env.FANZA_AFFILIATE_ID;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Supabase URL or Anon Key is not set.');
  process.exit(1);
}

if (!fanzaApiId || !fanzaAffiliateId) {
  console.error('FANZA_API_ID or FANZA_AFFILIATE_ID is not set in .env.local');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function fetchFanzaApiItems(queryParams: any) {
  const apiUrl = 'https://api.dmm.com/affiliate/v3/ItemList';
  const params = {
    api_id: fanzaApiId,
    affiliate_id: fanzaAffiliateId,
    site: 'FANZA',
    service: 'digital',
    floor: 'videoa',
    output: 'json',
    ...queryParams,
  };
  console.log('API Request Params:', JSON.stringify(params, null, 2)); // 追加

  try {
    const response = await axios.get(apiUrl, { params });
    console.log('FANZA API Raw Response:', JSON.stringify(response.data, null, 2)); // 追加
    return response.data.result;
  } catch (error: any) {
    console.error(`Error fetching data from FANZA API:`, error.message);
    if (error.response) {
      console.error('API Response Error:', error.response.data);
    }
    return null;
  }
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
    console.log(`Video with CID ${data.cid} already exists. Skipping insertion.`);
    return;
  }

  const { data: video, error: videoError } = await supabase
    .from('videos')
    .insert([
      {
        external_id: data.external_id, // external_idを追加
        title: data.title,
        thumbnail_url: data.thumbnail,
        sample_video_url: data.videoUrl,
        product_released_at: data.releaseDate,
        duration_seconds: data.duration,
        director: data.director,
        series: data.series,
        maker: data.maker,
        label: data.label,
        source: 'FANZA', // sourceを追加
        product_url: data.product_url, // product_urlを追加
      },
    ])
    .select();

  if (videoError) {
    console.error('Error inserting video:', videoError);
    return;
  }

  const videoId = video[0].id;

  // タグの挿入と関連付け
  for (const genreName of data.genres) {
    let { data: tag, error: tagError } = await supabase
      .from('tags')
      .select('id')
      .eq('name', genreName)
      .single();

    if (tagError && tagError.code === 'PGRST116') { // Not Found
      const { data: newTag, error: newTagError } = await supabase
        .from('tags')
        .insert([{ name: genreName }])
        .select();
      if (newTagError) {
        console.error('Error inserting new tag:', newTagError);
        continue;
      }
      tag = newTag[0];
    } else if (tagError) {
      console.error('Error fetching tag:', tagError);
      continue;
    }

    if (tag) {
      const { error: videoTagError } = await supabase
        .from('video_tags')
        .insert([{ video_id: videoId, tag_id: tag.id }]);
      if (videoTagError) {
        console.error('Error inserting video_tag:', videoTagError);
      }
    }
  }

  // 女優の挿入と関連付け
  for (const actressName of data.actresses) {
    let { data: performer, error: performerError } = await supabase
      .from('performers')
      .select('id')
      .eq('name', actressName)
      .single();

    if (performerError && performerError.code === 'PGRST116') { // Not Found
      const { data: newPerformer, error: newPerformerError } = await supabase
        .from('performers')
        .insert([{ name: actressName }])
        .select();
      if (newPerformerError) {
        console.error('Error inserting new performer:', newPerformerError);
        continue;
      }
      performer = newPerformer[0];
    } else if (performerError) {
      console.error('Error fetching performer:', performerError);
      continue;
    }

    if (performer) {
      const { error: videoPerformerError } = await supabase
        .from('video_performers')
        .insert([{ video_id: videoId, performer_id: performer.id }]);
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
      // 収録時間のパース (例: "120分" -> "120")
      const durationMatch = item.iteminfo?.size?.match(/(\d+)分/);
      const duration = durationMatch ? durationMatch[1] : null;

      // ジャンルと女優の抽出
      const genres = item.iteminfo?.genre ? (Array.isArray(item.iteminfo.genre) ? item.iteminfo.genre.map((g: any) => g.name) : [item.iteminfo.genre.name]) : [];
      const actresses = item.iteminfo?.actress ? (Array.isArray(item.iteminfo.actress) ? item.iteminfo.actress.map((a: any) => a.name) : [item.iteminfo.actress.name]) : [];

      const videoData = {
        external_id: item.content_id, // external_idを追加
        title: item.title,
        thumbnail: item.imageURL?.large || item.imageURL?.list,
        videoUrl: 'https://example.com/sample.mp4', // 仮のURL、APIからは直接取得できないため
        releaseDate: item.date,
        duration: duration,
        director: item.iteminfo?.director?.name,
        series: item.iteminfo?.series?.name,
        maker: item.iteminfo?.maker?.name,
        label: item.iteminfo?.label?.name,
        genres: genres,
        actresses: actresses,
        product_url: item.URL, // product_urlを追加
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
