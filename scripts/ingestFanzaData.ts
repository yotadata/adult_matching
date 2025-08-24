require('dotenv').config({ path: './.env.local' });
import { createClient } from '@supabase/supabase-js';
import { mapFanzaItem, VideoRecord } from './fanza_ingest'; // fanza_ingest.tsからインポート

// 環境変数の読み込み
const FANZA_API_ID = process.env.FANZA_API_ID;
const FANZA_AFFILIATE_ID = process.env.FANZA_AFFILIATE_ID;
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!FANZA_API_ID || !FANZA_AFFILIATE_ID) {
  console.error('FANZA_API_ID and FANZA_AFFILIATE_ID must be set in environment variables.');
  process.exit(1);
}

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY must be set in environment variables.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

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
    console.log('FANZA Floor API Response:', JSON.stringify(data, null, 2));
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
    console.log('FANZA API Response:', JSON.stringify(data, null, 2));
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

    console.log(`Fetched ${videoRecords.length} items from FANZA (offset: ${offset}).`);

    for (let i = 0; i < fanzaItems.length; i++) {
      const record = videoRecords[i];
      const originalItem = fanzaItems[i]; // 対応する元のアイテムを取得

      // videos テーブルに upsert するレコードから performers と tags を除外
      const videoRecordForUpsert: Omit<VideoRecord, 'performers' | 'tags'> = record;

      try { // forループ内のtry
        const { data, error } = await supabase
          .from('videos')
          .upsert(videoRecordForUpsert, { onConflict: 'source, distribution_code, maker_code' }) as { data: any[] | null, error: any };

        if (error) {
          // 重複エラーはスキップ
          if (error.code === '23505') { // unique_violation
            // console.warn(`Skipping duplicate record: ${record.title}`);
          } else {
            console.error('Error inserting record:', error);
          }
        } else {
          insertedCount++;
          if (data && data.length > 0) { // dataがnullでなく、かつ要素があることを確認
            const videoId = data[0].id; // upsert成功時に返される動画のIDを取得

            // 女優の処理
            if (originalItem && originalItem.iteminfo && originalItem.iteminfo.actress) {
              const actresses = originalItem.iteminfo.actress;
              for (const actress of actresses) {
                // performers テーブルに upsert
                const { data: performerData, error: performerError } = await supabase
                  .from('performers')
                  .upsert({ name: actress.name, fanza_actress_id: actress.id }, { onConflict: 'fanza_actress_id' })
                  .select('id'); // 挿入または更新されたperformerのIDを取得

                if (performerError) {
                  console.error('Error upserting performer:', performerError);
                  continue;
                }

                const performerId = performerData[0].id;

                // video_performers テーブルに挿入
                const { error: videoPerformerError } = await supabase
                  .from('video_performers')
                  .insert({ video_id: videoId, performer_id: performerId });

                if (videoPerformerError) {
                  // 重複エラーはスキップ
                  if (videoPerformerError.code === '23505') { // unique_violation
                    // console.warn(`Skipping duplicate video_performer entry for video ${videoId} and performer ${performerId}`);
                  } else {
                    console.error('Error inserting video_performer:', videoPerformerError);
                  }
                }
              }
            }

            // タグの処理
            if (originalItem && originalItem.iteminfo && originalItem.iteminfo.genre) {
              const genres = originalItem.iteminfo.genre;
              for (const genre of genres) {
                // tags テーブルに upsert
                const { data: tagData, error: tagError } = await supabase
                  .from('tags')
                  .upsert({ name: genre.name }, { onConflict: 'name' })
                  .select('id');

                if (tagError) {
                  console.error('Error upserting tag:', tagError);
                  continue;
                }

                const tagId = tagData[0].id;

                // video_tags テーブルに挿入
                const { error: videoTagError } = await supabase
                  .from('video_tags')
                  .insert({ video_id: videoId, tag_id: tagId });

                if (videoTagError) {
                  if (videoTagError.code === '23505') {
                    // console.warn(`Skipping duplicate video_tag entry for video ${videoId} and tag ${tagId}`);
                  } else {
                    console.error('Error inserting video_tag:', videoTagError);
                  }
                }
              }
            }

          } else { // if (data && data.length > 0) の else
            console.warn('Upsert successful but no data returned for video:', record.title);
          }
        } // if (error) の else の閉じ括弧
      } catch (e) { // forループ内のcatch
        console.error('Unexpected error during upsert:', e);
      } // try の閉じ括弧
    } // forループの閉じ括弧

    totalCount += videoRecords.length;
    console.log(`Processed ${totalCount} items. Inserted/Updated: ${insertedCount}`);

    // 次のページがあるか確認
    if (data.result.total_count && (offset + hits) <= data.result.total_count) {
      offset += hits;
    } else {
      break;
    }
  }

  console.log(`FANZA data ingestion complete. Total processed: ${totalCount}, Successfully inserted/updated: ${insertedCount}`);
}

ingestFanzaData();