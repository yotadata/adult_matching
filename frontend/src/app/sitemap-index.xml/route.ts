// Next.js の generateSitemaps では /sitemap.xml の自動インデックスが
// 環境によって機能しないため、手動でインデックスを返す
export const revalidate = 86400;

const SITE_URL = 'https://www.seihekilab.com';

export async function GET() {
  const sitemaps = ['0', '1', '2', '3'];
  const lastmod = new Date().toISOString();

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemaps.map((id) => `  <sitemap>
    <loc>${SITE_URL}/sitemap/${id}.xml</loc>
    <lastmod>${lastmod}</lastmod>
  </sitemap>`).join('\n')}
</sitemapindex>`;

  return new Response(xml, {
    headers: {
      'Content-Type': 'application/xml',
      'Cache-Control': 's-maxage=86400, stale-while-revalidate',
    },
  });
}
