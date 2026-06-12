const FANZA_AFFILIATE_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';

/**
 * ソースに応じたサンプル動画の埋め込みURLを返す。
 * - mgs: sample_video_url (MP4) をそのまま返す → <video> タグで再生
 * - FANZA系: DMM litevideo iframe URL を返す → <iframe> で再生
 */
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
