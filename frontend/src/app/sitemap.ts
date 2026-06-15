import { MetadataRoute } from 'next';
import { supabaseAdmin } from '@/lib/supabaseAdmin';

// 24時間ごとに再生成
export const revalidate = 86400;
// ビルド時の静的生成をスキップ（DBアクセスによるタイムアウト防止）
export const dynamic = 'force-dynamic';

const SITE_URL = 'https://www.seihekilab.com';
const PAGE_SIZE = 1000;

// サイトマップを4分割してタイムアウトを防ぐ
// /sitemap/0 = 静的ページ
// /sitemap/1 = 動画ページ
// /sitemap/2 = タグページ
// /sitemap/3 = 女優ページ
export function generateSitemaps() {
  return [{ id: '0' }, { id: '1' }, { id: '2' }, { id: '3' }];
}

export default async function sitemap({ id }: { id: string }): Promise<MetadataRoute.Sitemap> {
  // 静的ページ
  if (id === '0') {
    return [
      { url: `${SITE_URL}/`, lastModified: new Date(), changeFrequency: 'daily', priority: 1.0 },
      { url: `${SITE_URL}/grid`, lastModified: new Date(), changeFrequency: 'daily', priority: 0.9 },
      { url: `${SITE_URL}/swipe`, lastModified: new Date(), changeFrequency: 'daily', priority: 0.8 },
      { url: `${SITE_URL}/quiz`, lastModified: new Date(), changeFrequency: 'monthly', priority: 0.7 },
      { url: `${SITE_URL}/quiz/characters`, lastModified: new Date(), changeFrequency: 'monthly', priority: 0.6 },
      { url: `${SITE_URL}/tags`, lastModified: new Date(), changeFrequency: 'weekly', priority: 0.8 },
      { url: `${SITE_URL}/performers`, lastModified: new Date(), changeFrequency: 'weekly', priority: 0.8 },
    ];
  }

  // 動画ページ
  if (id === '1') {
    const entries: MetadataRoute.Sitemap = [];
    let page = 0;
    while (true) {
      const from = page * PAGE_SIZE;
      const { data, error } = await supabaseAdmin
        .from('videos')
        .select('id, created_at')
        .order('created_at', { ascending: false })
        .range(from, from + PAGE_SIZE - 1);
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
    return entries;
  }

  // タグページ
  if (id === '2') {
    const entries: MetadataRoute.Sitemap = [];
    let page = 0;
    while (true) {
      const from = page * PAGE_SIZE;
      const { data, error } = await supabaseAdmin
        .from('tags')
        .select('id')
        .range(from, from + PAGE_SIZE - 1);
      if (error || !data || data.length === 0) break;
      for (const t of data) {
        entries.push({
          url: `${SITE_URL}/tags/${t.id}`,
          lastModified: new Date(),
          changeFrequency: 'weekly',
          priority: 0.8,
        });
      }
      if (data.length < PAGE_SIZE) break;
      page++;
    }
    return entries;
  }

  // 女優ページ
  if (id === '3') {
    const entries: MetadataRoute.Sitemap = [];
    let page = 0;
    while (true) {
      const from = page * PAGE_SIZE;
      const { data, error } = await supabaseAdmin
        .from('performers')
        .select('id')
        .range(from, from + PAGE_SIZE - 1);
      if (error || !data || data.length === 0) break;
      for (const p of data) {
        entries.push({
          url: `${SITE_URL}/performers/${p.id}`,
          lastModified: new Date(),
          changeFrequency: 'weekly',
          priority: 0.7,
        });
      }
      if (data.length < PAGE_SIZE) break;
      page++;
    }
    return entries;
  }

  return [];
}
