import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

export type Video = {
  id: string; // uuid
  title: string;
  description?: string | null;
  external_id?: string | null;
  product_url?: string | null;
  thumbnail_url?: string | null;
};

export type AiRecommendResponse = {
  personalized: Video[];
  trending: Video[];
};

export function useAiRecommend() {
  const [data, setData] = useState<AiRecommendResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data: { session } } = await supabase.auth.getSession();
        const headers: HeadersInit = {};
        if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;
        const { data: body, error } = await supabase.functions.invoke('ai-recommend', { headers });
        if (error) throw error;
        setData(body as AiRecommendResponse);
      } catch (e: any) {
        setError(e?.message || '取得に失敗しました');
      } finally {
        setLoading(false);
      }
    };
    run();
  }, []);

  return { data, loading, error };
}
