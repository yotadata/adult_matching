import { MetadataRoute } from 'next';
import { supabase } from '@/lib/supabase';

// 24時間ごとに再生成
export const revalidate = 86400;

const SITE_URL = 'https://www.seihekilab.com';

const PAGE_SIZE = 1000;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const entries: MetadataRoute.Sitemap = [
    { url: `${SITE_URL}/grid`, lastModified: new Date(), changeFrequency: 'daily', priority: 1.0 },
    { url: `${SITE_URL}/swipe`, lastModified: new Date(), changeFrequency: 'daily', priority: 0.9 },
    { url: `${SITE_URL}/about`, lastModified: new Date(), changeFrequency: 'monthly', priority: 0.8 },
    { url: `${SITE_URL}/quiz`, lastModified: new Date(), changeFrequency: 'monthly', priority: 0.7 },
    { url: `${SITE_URL}/quiz/characters`, lastModified: new Date(), changeFrequency: 'monthly', priority: 0.6 },
    { url: `${SITE_URL}/tags`, lastModified: new Date(), changeFrequency: 'weekly', priority: 0.8 },
  ];

  // 作品LPを全件追加
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

  // タグLPを全件追加（ページネーション）
  let tagPage = 0;
  while (true) {
    const from = tagPage * PAGE_SIZE;
    const { data: tags, error } = await supabase
      .from('tags')
      .select('id')
      .range(from, from + PAGE_SIZE - 1);
    if (error || !tags || tags.length === 0) break;
    for (const t of tags) {
      entries.push({
        url: `${SITE_URL}/tags/${t.id}`,
        lastModified: new Date(),
        changeFrequency: 'weekly',
        priority: 0.8,
      });
    }
    if (tags.length < PAGE_SIZE) break;
    tagPage++;
  }

  // 女優LPを全件追加（ページネーション）
  let performerPage = 0;
  while (true) {
    const from = performerPage * PAGE_SIZE;
    const { data: performers, error } = await supabase
      .from('performers')
      .select('id')
      .range(from, from + PAGE_SIZE - 1);
    if (error || !performers || performers.length === 0) break;
    for (const p of performers) {
      entries.push({
        url: `${SITE_URL}/performers/${p.id}`,
        lastModified: new Date(),
        changeFrequency: 'weekly',
        priority: 0.7,
      });
    }
    if (performers.length < PAGE_SIZE) break;
    performerPage++;
  }

  return entries;
}
