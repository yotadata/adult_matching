const FANZA_AFFILIATE_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';
const MGS_AFFILIATE_ID   = process.env.NEXT_PUBLIC_MGS_AFFILIATE_ID   ?? 'HU3ADNBETQPYWHO8EFF88GY3NH';

export function buildMgsAffiliateUrl(raw: string, mgsAfId: string): string {
  try {
    const u = new URL(raw);
    u.searchParams.set('agef', '1');
    u.searchParams.set('utm_medium', 'mgs_affiliate');
    u.searchParams.set('utm_source', 'mgs_affiliate_linktool');
    u.searchParams.set('utm_campaign', 'mgs_affiliate_linktool');
    u.searchParams.set('utm_content', mgsAfId);
    u.searchParams.set('form', `mgs_asp_linktool_${mgsAfId}`);
    return u.toString();
  } catch {
    return raw;
  }
}

/**
 * 外部購入リンクを解決する。
 * - affiliate_url が設定済みならそのまま使う（MGS含む全ソース）
 * - MGS: product_url に MGS アフィリエイトパラメータを付与
 * - FANZA系: al.fanza.co.jp でラップ
 */
export function resolveProductUrl({
  source,
  productUrl,
  affiliateUrl,
  mgsAfId,
  fanzaAfId,
}: {
  source?: string | null;
  productUrl?: string | null;
  affiliateUrl?: string | null;
  mgsAfId?: string | null;
  fanzaAfId?: string | null;
}): string {
  if (affiliateUrl) return affiliateUrl;
  if (!productUrl) return '';

  if (source === 'mgs') {
    return buildMgsAffiliateUrl(productUrl, mgsAfId ?? MGS_AFFILIATE_ID);
  }

  const afId = fanzaAfId ?? FANZA_AFFILIATE_ID;
  if (productUrl.startsWith('https://al.fanza.co.jp/')) return productUrl;
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(productUrl)}&af_id=${encodeURIComponent(afId)}&ch=link_tool&ch_id=link`;
}

export function resolveEmbedUrl({
  source,
  externalId,
  sampleVideoUrl,
}: {
  source?: string | null;
  externalId?: string | null;
  sampleVideoUrl?: string | null;
}): { type: 'mp4'; url: string } | { type: 'iframe'; url: string } | null {
  if (source === 'mgs') {
    if (!sampleVideoUrl) return null;
    return { type: 'mp4', url: sampleVideoUrl };
  }
  if (!externalId) return null;
  return {
    type: 'iframe',
    url: `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${FANZA_AFFILIATE_ID}/cid=${externalId}/size=1280_720/`,
  };
}

const VR_TAG_NAMES = ['VR', 'VR対応', 'VR専用', 'ハイクオリティVR', '8KVR'];

export const isVrContent = (tags?: { name: string }[] | null): boolean =>
  tags?.some((t) => VR_TAG_NAMES.includes(t.name)) ?? false;

export const resolveFanzaVrUrl = ({
  productUrl,
  externalId,
}: {
  productUrl?: string | null;
  externalId?: string | null;
}): string => {
  if (productUrl) return productUrl;
  if (externalId) return `https://video.dmm.co.jp/av/content/?id=${externalId}`;
  return '';
};

export const isUpcomingRelease = (value?: string | null): boolean => {
  if (!value) return false;
  const releaseDate = new Date(value);
  if (Number.isNaN(releaseDate.getTime())) return false;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const releaseDay = new Date(releaseDate);
  releaseDay.setHours(0, 0, 0, 0);
  return releaseDay.getTime() >= today.getTime();
};

export {};
