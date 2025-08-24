// Shared types for Supabase Edge Functions

export interface CorsHeaders {
  'Access-Control-Allow-Origin': string;
  'Access-Control-Allow-Headers': string;
  'Access-Control-Allow-Methods': string;
}

export interface VideoData {
  id: string;
  title: string;
  description: string;
  duration_seconds: number;
  thumbnail_url: string;
  preview_video_url: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}