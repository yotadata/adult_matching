const NOW_PRINTING_URL = 'now_printing';

function isNowPrinting(url: string | null | undefined): boolean {
  return !url || url.includes(NOW_PRINTING_URL);
}

/**
 * pics.dmm.co.jp の js-NNN URL を awsimgsrc の大サイズ jp-NNN URL に変換する。
 * FANZA 素人動画は thumbnail_url が now_printing になるケースが多いため、
 * image_urls[0] から大サイズ画像URLを生成してサムネイルとして使う。
 */
function toAmateurLargeUrl(jsUrl: string): string {
  return jsUrl
    .replace('pics.dmm.co.jp/', 'awsimgsrc.dmm.co.jp/pics_dig/')
    .replace('/js-', '/jp-')
    + '?q=85';
}

/**
 * 動画データから表示用サムネイルURLを返す。
 * FANZA_AMATEUR かつ thumbnail_url が now_printing の場合、
 * image_urls[0] から生成した大サイズURLを primary、元の js URL を fallback として返す。
 */
export function resolveThumbnail(params: {
  source?: string | null;
  thumbnail_url?: string | null;
  image_urls?: string[] | null;
}): { primary: string | null; fallback: string | null } {
  const { source, thumbnail_url, image_urls } = params;

  if (source === 'FANZA_AMATEUR' && isNowPrinting(thumbnail_url) && image_urls?.[0]) {
    const jsUrl = image_urls[0];
    return {
      primary: toAmateurLargeUrl(jsUrl),
      fallback: jsUrl,
    };
  }

  return { primary: thumbnail_url ?? null, fallback: null };
}
