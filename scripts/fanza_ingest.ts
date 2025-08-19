export interface VideoRecord {
  external_id: string;
  title: string;
  description?: string;
  categories?: string[];
  performers?: string[];
  duration_seconds?: number;
  thumbnail_url?: string;
  preview_video_url?: string;
  distribution_code?: string;
  maker_code?: string;
  director?: string;
  series?: string;
  maker?: string;
  label?: string;
  genre?: string;
  price?: number;
  distribution_started_at?: string;
  product_released_at?: string;
  sample_video_url?: string;
  image_urls?: string[];
  source: string;
  published_at?: string;
}

export function mapFanzaItem(item: any): VideoRecord {
  return {
    external_id: item.product_id,
    title: item.title,
    description: item.description,
    categories: item.categories,
    performers: item.cast,
    duration_seconds: item.duration,
    thumbnail_url: item.thumbnail,
    preview_video_url: item.preview,
    distribution_code: item.dvd_id,
    maker_code: item.maker_id,
    director: item.director,
    series: item.series,
    maker: item.maker,
    label: item.label,
    genre: item.genre,
    price: item.price,
    distribution_started_at: item.sale_start,
    product_released_at: item.release_date,
    sample_video_url: item.sample_movie,
    image_urls: item.images,
    source: 'fanza',
    published_at: item.release_date,
  };
}
