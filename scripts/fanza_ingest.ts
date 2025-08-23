export interface VideoRecord {
  external_id: string;
  title: string;
  description?: string;
  duration_seconds?: number;
  thumbnail_url?: string;
  preview_video_url?: string;
  distribution_code?: string;
  maker_code?: string;
  director?: string;
  series?: string;
  maker?: string;
  label?: string;
  genre?: string[];
  price?: number;
  distribution_started_at?: string;
  product_released_at?: string;
  sample_video_url?: string;
  image_urls?: string[];
  source: string;
  published_at?: string;
}

export function mapFanzaItem(item: any): VideoRecord {
  // item.iteminfo が存在しない場合の考慮
  const iteminfo = item.iteminfo || {};

  // 配列フィールドの処理ヘルパー
  const mapNames = (arr: any[] | undefined) => arr ? arr.map(i => i.name) : undefined;
  const mapImages = (obj: any) => {
    const urls: string[] = [];
    if (obj && obj.list) urls.push(obj.list);
    if (obj && obj.small) urls.push(obj.small);
    if (obj && obj.large) urls.push(obj.large);
    return urls.length > 0 ? urls : undefined;
  };
  const mapSampleImages = (obj: any) => {
    const urls: string[] = [];
    if (obj && obj.sample_s && Array.isArray(obj.sample_s.image)) {
      urls.push(...obj.sample_s.image);
    }
    if (obj && obj.sample_l && Array.isArray(obj.sample_l.image)) {
      urls.push(...obj.sample_l.image);
    }
    return urls.length > 0 ? urls : undefined;
  };

  return {
    external_id: item.product_id,
    title: item.title,
    description: item.description, // JSONにdescriptionフィールドがないため、これはundefinedになる
    duration_seconds: item.duration, // JSONにdurationフィールドがないため、これはundefinedになる
    thumbnail_url: item.imageURL?.large || item.imageURL?.list, // largeがなければlist
    preview_video_url: item.sampleMovieURL?.size_476_306, // JSONにpreviewフィールドがないため、これはundefinedになる
    distribution_code: item.content_id, // dvd_idはcontent_idにマッピング
    maker_code: iteminfo.maker?.[0]?.id, // maker_idはiteminfo.makerのidにマッピング
    director: mapNames(iteminfo.director)?.[0], // directorは配列の最初の要素
    series: mapNames(iteminfo.series)?.[0], // seriesは配列の最初の要素
    maker: mapNames(iteminfo.maker)?.[0], // makerは配列の最初の要素
    label: mapNames(iteminfo.label)?.[0], // labelは配列の最初の要素
    genre: mapNames(iteminfo.genre), // genreは配列
    price: item.prices?.price ? parseFloat(item.prices.price.replace('~', '')) : undefined, // priceは文字列なので数値に変換
    distribution_started_at: item.date, // sale_startはdateにマッピング
    product_released_at: item.date, // release_dateはdateにマッピング
    sample_video_url: item.sampleMovieURL?.size_476_306, // sample_movieはsampleMovieURLにマッピング
    image_urls: [...(mapImages(item.imageURL) || []), ...(mapSampleImages(item.sampleImageURL) || [])], // imageURLとsampleImageURLを結合
    source: 'fanza',
    published_at: item.date, // published_atはdateにマッピング
  };
}
