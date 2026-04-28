import { MetadataRoute } from 'next';

const SITE_URL = 'https://seihekilab.com';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        allow: '/',
        disallow: [
          '/api/',
          '/account-management',
          '/contact',
        ],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
