const FANZA_AFFILIATE_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';

/**
 * ソースに応じたサンプル動画の埋め込みURLを返す。
 * - mgs: sample_video_url (MP4) をそのまま返す → <video> タグで再生
 * - FANZA系: DMM litevideo iframe URL を返す → <iframe> で再生
 */
/**
 * 外部購入リンクを解決する。
 * - affiliate_url が設定されている場合はそれを優先（MGSはここに ?aff= 付きURLが入る）
 * - FANZA系かつ affiliate_url がない場合のみ al.fanza.co.jp でラップする
 */
export function resolveProductUrl({
  source,
  productUrl,
  affiliateUrl,
}: {
  source?: string | null;
  productUrl?: string | null;
  affiliateUrl?: string | null;
}): string {
  // affiliate_url が設定済みならそのまま使う（MGS含む全ソース）
  if (affiliateUrl) return affiliateUrl;
  if (!productUrl) return '';

  // FANZAのみラッパーを適用
  if (source === 'mgs') return productUrl;
  const afId = FANZA_AFFILIATE_ID;
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
