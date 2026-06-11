/**
 * pics.dmm.co.jp の js-NNN URL を awsimgsrc の大サイズ jp-NNN URL に変換する。
 */
function toAmateurLargeUrl(jsUrl: string): string {
  return jsUrl
    .replace('pics.dmm.co.jp/', 'awsimgsrc.dmm.co.jp/pics_dig/')
    .replace('js-', 'jp-')
    + '?q=85';
}

/**
 * 動画データから表示用サムネイルURLを返す。
 * FANZA_AMATEUR は thumbnail_url が now_printing にリダイレクトされるケースが多いため、
 * image_urls[0] から生成した大サイズ（jp）URLを primary、元の js URL を fallback として返す。
 */
export function resolveThumbnail(params: {
  source?: string | null;
  thumbnail_url?: string | null;
  image_urls?: string[] | null;
}): { primary: string | null; fallback: string | null } {
  const { source, thumbnail_url, image_urls } = params;

  if (source === 'FANZA_AMATEUR' && image_urls?.[0]) {
    const jsUrl = image_urls[0];
    return {
      primary: toAmateurLargeUrl(jsUrl),
      fallback: jsUrl,
    };
  }

  return { primary: thumbnail_url ?? null, fallback: null };
}
