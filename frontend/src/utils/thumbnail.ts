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
 * FANZA_AMATEUR の thumbnail_url（jm.jpg）を jp-001.jpg（大サイズ）に変換する。
 * image_urls が未取得の場合のフォールバック。
 */
function amateurThumbToLarge(thumbnailUrl: string): string {
  // gerk657jm.jpg → gerk657jp-001.jpg
  return thumbnailUrl.replace(/([^/]+)jm\.jpg$/, '$1jp-001.jpg');
}

/**
 * 動画データから表示用サムネイルURLを返す。
 * FANZA_AMATEUR は thumbnail_url が now_printing にリダイレクトされるケースが多いため、
 * image_urls[0] から生成した大サイズ（jp）URLを primary、元の js URL を fallback として返す。
 * image_urls が null の場合は thumbnail_url の jm→jp-001 変換を試みる。
 */
export function resolveThumbnail(params: {
  source?: string | null;
  thumbnail_url?: string | null;
  image_urls?: string[] | null;
}): { primary: string | null; fallback: string | null } {
  const { source, thumbnail_url, image_urls } = params;

  if (source === 'FANZA_AMATEUR') {
    if (image_urls?.[0]) {
      const jsUrl = image_urls[0];
      return {
        primary: toAmateurLargeUrl(jsUrl),
        // thumbnail_url (jm.jpg) を最終フォールバックとして使う。
        // js-001.jpg はサンプル画像がない動画では実在しないため使わない。
        fallback: thumbnail_url ?? jsUrl,
      };
    }
    // image_urls が未取得の場合、thumbnail_url の jm→jp-001 変換を試みる
    if (thumbnail_url?.includes('jm.jpg')) {
      return {
        primary: amateurThumbToLarge(thumbnail_url),
        fallback: thumbnail_url,
      };
    }
  }

  return { primary: thumbnail_url ?? null, fallback: null };
}
