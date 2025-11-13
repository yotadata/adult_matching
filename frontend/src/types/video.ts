export interface Video {
  external_id: string;
  title: string;
  thumbnail_url: string | null;
  price: number | null;
  product_released_at: string | null;
  tags: { id: string; name: string }[] | null;
  performers: { id: string; name: string }[] | null;
  product_url: string | null;
}
