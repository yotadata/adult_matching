import { useCallback, useEffect, useMemo, useState } from 'react';
import { supabase } from '@/lib/supabase';

export type CustomIntent = {
  duration?: 'short' | 'medium' | 'long';
  mood?: 'sweet' | 'passion' | 'healing' | 'curious';
  context?: 'solo' | 'partner' | 'restricted';
};

export type AiRecommendSectionItem = {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  product_url: string | null | undefined;
  sample_video_url: string | null;
  preview_video_url: string | null | undefined;
  tags: Array<{ id: string; name: string }>;
  performers: Array<{ id: string; name: string }>;
  duration_minutes: number | null;
  metrics: {
    score?: number | null;
    popularity_score?: number | null;
    product_released_at?: string | null;
    source: string;
  };
  reason: {
    summary: string;
    detail: string;
    highlights: string[];
  };
};

export type AiRecommendSection = {
  id: string;
  title: string;
  rationale: string;
  items: AiRecommendSectionItem[];
};

export type AiRecommendResponse = {
  generated_at: string;
  mode: {
    id: string;
    label: string;
    description: string;
    rationale: string;
    custom_intent: CustomIntent;
  };
  sections: AiRecommendSection[];
  metadata: {
    personalized_candidates: number;
    trending_candidates: number;
    fresh_candidates: number;
    has_user_context: boolean;
    limit_per_section: number;
  };
};

type UseAiRecommendOptions = {
  modeId: string;
  customIntent?: CustomIntent;
  limitPerSection?: number;
};

export function useAiRecommend(options: UseAiRecommendOptions) {
  const { modeId, customIntent, limitPerSection } = options;
  const [data, setData] = useState<AiRecommendResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const payload = useMemo(() => ({
    mode_id: modeId,
    custom_intent: customIntent ?? undefined,
    limit_per_section: limitPerSection,
  }), [modeId, customIntent, limitPerSection]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      const headers: HeadersInit = {};
      if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;
      const { data: body, error } = await supabase.functions.invoke<AiRecommendResponse>('ai-recommend', {
        headers,
        body: payload,
      });
      if (error) throw error;
      setData(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'レコメンドの取得に失敗しました');
    } finally {
      setLoading(false);
    }
  }, [payload]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
}
