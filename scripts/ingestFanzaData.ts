require('dotenv').config({ path: './.env.local' });
import { createClient } from '@supabase/supabase-js';
import { mapFanzaItem, VideoRecord } from './fanza_ingest'; // fanza_ingest.tsからインポート

// 環境変数の読み込み
const FANZA_API_ID = process.env.FANZA_API_ID;
const FANZA_AFFILIATE_ID = process.env.FANZA_AFFILIATE_ID;
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!FANZA_API_ID || !FANZA_AFFILIATE_ID) {
  console.error('FANZA_API_ID and FANZA_AFFILIATE_ID must be set in environment variables.');
  process.exit(1);
}

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

const FANZA_API_BASE_URL = 'https://api.dmm.com/affiliate/v3/ItemList';
const FANZA_FLOOR_API_URL = 'https://api.dmm.com/affiliate/v3/FloorList';

async function fetchFanzaFloors(): Promise<any> {
  const params = new URLSearchParams({
    api_id: FANZA_API_ID!,
    affiliate_id: FANZA_AFFILIATE_ID!,
    output: 'json',
  });

  const url = `${FANZA_FLOOR_API_URL}?${params.toString()}`;

  try {
    console.log(`Fetching floor data from: ${url}`);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    
    return data;
  } catch (error) {
    console.error('Error fetching FANZA floor data:', error);
    return null;
  }
}

async function fetchFanzaData(offset: number = 1, hits: number = 100): Promise<any> {
  // 現在の日付から1年前の日付を計算
  const oneYearAgo = new Date();
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
  const year = oneYearAgo.getFullYear();
  const month = (oneYearAgo.getMonth() + 1).toString().padStart(2, '0');
  const day = oneYearAgo.getDate().toString().padStart(2, '0');
  const dateFilter = `${year}-${month}-${day}`;

  const params = new URLSearchParams({
    api_id: FANZA_API_ID!,
    affiliate_id: FANZA_AFFILIATE_ID!,
    site: 'FANZA', // FANZAサイトを指定
    service: 'digital', // デジタルコンテンツ
    floor: 'videoa', // アダルトビデオ
    hits: hits.toString(),
    offset: offset.toString(),
    sort: 'rank', // ランキング順
    output: 'json',
    // 追加: 発売日フィルタ
    date: dateFilter,
  });

  const url = `${FANZA_API_BASE_URL}?${params.toString()}`;

  try {
    console.log(`Fetching data from: ${url}`);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    
    return data;
  } catch (error) {
    console.error('Error fetching FANZA data:', error);
    return null;
  }
}

async function ingestFanzaData() {
  // 追加: フロア情報を取得
  const floorData = await fetchFanzaFloors();
  if (!floorData || !floorData.result || !floorData.result.floors) {
    console.error('Failed to fetch floor data. Exiting.');
  }

  let offset = 1;
  const hits = 100; // 1回のリクエストで取得する件数
  let totalCount = 0;
  let insertedCount = 0;

  while (true) {
    const data: any = await fetchFanzaData(offset, hits);

    if (!data || !data.result || !data.result.items || data.result.items.length === 0) {
      console.log('No more data to fetch or an error occurred.');
      break;
    }

    const fanzaItems = data.result.items;
    const videoRecords: VideoRecord[] = fanzaItems.map(mapFanzaItem);

    

    // Step 1: Collect video records for batch upsert.
    const videosToUpsert: Omit<VideoRecord, 'performers' | 'tags'>[] = videoRecords;

    // Step 2: Perform batch upsert for videos.
    const { data: upsertedVideos, error: videosUpsertError } = await supabase
      .from('videos')
      .upsert(videosToUpsert, { onConflict: 'source, distribution_code, maker_code' })
      .select('id, source, distribution_code, maker_code');

    if (videosUpsertError) {
      console.error('Error batch upserting videos:', videosUpsertError);
      // エラーが発生しても処理を続行するか、ここでbreakするかは要検討
      // 今回は続行するが、エラーの種類によってはbreakも考慮
    } else {
      insertedCount += (upsertedVideos?.length || 0); // 挿入/更新された数を加算
    }

    // Step 3: Map video_ids for linking performers and tags.
    const videoIdMap = new Map<string, string>(); // key: source_distribution_code_maker_code, value: video_id
    upsertedVideos?.forEach(video => {
      const key = `${video.source}_${video.distribution_code}_${video.maker_code}`;
      videoIdMap.set(key, video.id);
    });

    // Step 4: Collect unique performers and tags for batch upsert.
    const allPerformers: { name: string; fanza_actress_id: string }[] = [];
    const allTags: { name: string }[] = [];
    const videoPerformersToInsert: { video_id: string; performer_id: string }[] = [];
    const videoTagsToInsert: { video_id: string; tag_id: string }[] = [];

    const uniquePerformers = new Map<string, { name: string; fanza_actress_id: string }>();
    const uniqueTags = new Map<string, { name: string }>();

    fanzaItems.forEach((originalItem: any) => {
      const videoId = videoIdMap.get(`${originalItem.source}_${originalItem.content_id}_${originalItem.iteminfo?.maker?.[0]?.id}`);

      if (videoId) {
        // 女優の処理
        if (originalItem.iteminfo && originalItem.iteminfo.actress) {
          originalItem.iteminfo.actress.forEach((actress: any) => {
            if (!uniquePerformers.has(actress.id)) {
              uniquePerformers.set(actress.id, { name: actress.name, fanza_actress_id: actress.id });
            }
          });
        }

        // タグの処理
        if (originalItem.iteminfo && originalItem.iteminfo.genre) {
          originalItem.iteminfo.genre.forEach((genre: any) => {
            if (!uniqueTags.has(genre.name)) {
              uniqueTags.set(genre.name, { name: genre.name });
            }
          });
        }
      }
    });

    allPerformers.push(...Array.from(uniquePerformers.values()));
    allTags.push(...Array.from(uniqueTags.values()));

    // Step 5: Perform batch upsert for performers and tags.
    const { data: upsertedPerformers, error: performersUpsertError } = await supabase
      .from('performers')
      .upsert(allPerformers, { onConflict: 'fanza_actress_id' })
      .select('id, fanza_actress_id');

    if (performersUpsertError) {
      console.error('Error batch upserting performers:', performersUpsertError);
    }

    const { data: upsertedTags, error: tagsUpsertError } = await supabase
      .from('tags')
      .upsert(allTags, { onConflict: 'name' })
      .select('id, name');

    if (tagsUpsertError) {
      console.error('Error batch upserting tags:', tagsUpsertError);
    }

    // Create maps for performer_id and tag_id
    const performerIdMap = new Map<string, string>(); // key: fanza_actress_id, value: performer_id
    upsertedPerformers?.forEach(p => performerIdMap.set(p.fanza_actress_id, p.id));

    const tagIdMap = new Map<string, string>(); // key: tag_name, value: tag_id
    upsertedTags?.forEach(t => tagIdMap.set(t.name, t.id));

    // Step 6: Collect and batch insert into join tables.
    fanzaItems.forEach((originalItem: any) => {
      const videoId = videoIdMap.get(`${originalItem.source}_${originalItem.content_id}_${originalItem.iteminfo?.maker?.[0]?.id}`);

      if (videoId) {
        // video_performers
        if (originalItem.iteminfo && originalItem.iteminfo.actress) {
          originalItem.iteminfo.actress.forEach((actress: any) => {
            const performerId = performerIdMap.get(actress.id);
            if (performerId) {
              videoPerformersToInsert.push({ video_id: videoId, performer_id: performerId });
            }
          });
        }

        // video_tags
        if (originalItem.iteminfo && originalItem.iteminfo.genre) {
          originalItem.iteminfo.genre.forEach((genre: any) => {
            const tagId = tagIdMap.get(genre.name);
            if (tagId) {
              videoTagsToInsert.push({ video_id: videoId, tag_id: tagId });
            }
          });
        }
      }
    });

    // Perform batch inserts for join tables
    if (videoPerformersToInsert.length > 0) {
      const { error: insertError } = await supabase
        .from('video_performers')
        .insert(videoPerformersToInsert)
        .ignoreDuplicates();
      if (insertError) console.error('Error inserting video_performers:', insertError);
    }

    if (videoTagsToInsert.length > 0) {
      const { error: insertError } = await supabase
        .from('video_tags')
        .insert(videoTagsToInsert)
        .ignoreDuplicates();
      if (insertError) console.error('Error inserting video_tags:', insertError);
    }

    totalCount += videoRecords.length;
    console.log(`Processed ${totalCount} items. Inserted/Updated: ${insertedCount}`);

    // 次のページがあるか確認
    if (data.result.total_count && (offset + hits) <= data.result.total_count) {
      offset += hits;
    } else {
      break;
    }
  }