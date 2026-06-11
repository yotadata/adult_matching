import { createClient } from '@supabase/supabase-js';
import ws from 'ws';
import * as dotenv from 'dotenv';
dotenv.config({ path: 'docker/env/prd.env' });

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
  { auth: { persistSession: false }, realtime: { transport: ws } }
);

async function run() {
  const FETCH_PAGE = 1000;
  const UPDATE_CHUNK = 100; // URLが長くなりすぎないよう100件ずつ
  let offset = 0;
  const allVideoIds = new Set<string>();

  console.log('Collecting video_ids from video_performers...');
  while (true) {
    const { data, error } = await supabase
      .from('video_performers')
      .select('video_id')
      .range(offset, offset + FETCH_PAGE - 1);

    if (error) { console.error(error); process.exit(1); }
    if (!data || data.length === 0) break;
    for (const row of data) allVideoIds.add(row.video_id);
    offset += FETCH_PAGE;
    if (data.length < FETCH_PAGE) break;
  }
  console.log(`Total distinct video_ids: ${allVideoIds.size}`);

  const ids = [...allVideoIds];
  let updated = 0;
  for (let i = 0; i < ids.length; i += UPDATE_CHUNK) {
    const chunk = ids.slice(i, i + UPDATE_CHUNK);
    const { error } = await supabase
      .from('videos')
      .update({ has_performer: true })
      .in('id', chunk);
    if (error) { console.error('update error:', error); process.exit(1); }
    updated += chunk.length;
    if (updated % 10000 === 0) console.log(`  updated ${updated}/${ids.length}...`);
  }
  console.log(`Done. Updated ${updated} videos.`);
}

run();
