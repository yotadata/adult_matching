import { MetadataRoute } from 'next';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://seihekilab.com';

// 1 回のクエリで取得する件数（Supabase の上限に合わせて分割）
const PAGE_SIZE = 1000;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const entries: MetadataRoute.Sitemap = [
    { url: SITE_URL, lastModified: new Date(), changeFrequency: 'daily', priority: 1.0 },
    { url: `${SITE_URL}/swipe`, lastModified: new Date(), changeFrequency: 'daily', priority: 0.9 },
    { url: `${SITE_URL}/about`, lastModified: new Date(), changeFrequency: 'monthly', priority: 0.5 },
    { url: `${SITE_URL}/performers`, lastModified: new Date(), changeFrequency: 'weekly', priority: 0.8 },
    { url: `${SITE_URL}/tags`, lastModified: new Date(), changeFrequency: 'weekly', priority: 0.8 },
  ];

  // 作品 LP を全件追加（ページネーションで大量件数に対応）
  let page = 0;
  while (true) {
    const from = page * PAGE_SIZE;
    const to = from + PAGE_SIZE - 1;

    const { data, error } = await supabase
      .from('videos')
      .select('id, created_at')
      .order('created_at', { ascending: false })
      .range(from, to);

    if (error || !data || data.length === 0) break;

    for (const v of data) {
      entries.push({
        url: `${SITE_URL}/videos/${v.id}`,
        lastModified: v.created_at ? new Date(v.created_at) : new Date(),
        changeFrequency: 'monthly',
        priority: 0.7,
      });
    }

    if (data.length < PAGE_SIZE) break;
    page++;
  }

  // 出演者 LP を全件追加
  const { data: performers } = await supabase
    .from('performers')
    .select('id');
  for (const p of performers ?? []) {
    entries.push({
      url: `${SITE_URL}/performers/${p.id}`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    });
  }

  // ジャンル LP を全件追加
  const { data: tags } = await supabase
    .from('tags')
    .select('id');
  for (const t of tags ?? []) {
    entries.push({
      url: `${SITE_URL}/tags/${t.id}`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    });
  }

  return entries;
}
