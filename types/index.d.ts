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
  price?: number;
  distribution_started_at?: string;
  product_released_at?: string;
  sample_video_url?: string;
  image_urls?: string[];
  source: string;
  published_at?: string;
  product_url?: string;
}
