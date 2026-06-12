export interface Video {
  external_id: string;
  title: string;
  thumbnail_url: string | null;
  sample_image_urls: string[] | null;
  author: string | null;
  price: number | null;
  page_count: number | null;
  product_released_at: string | null;
  tags: { id: string; name: string }[] | null;
  product_url: string | null;
  affiliate_url: string | null;
  source?: string | null;
}
