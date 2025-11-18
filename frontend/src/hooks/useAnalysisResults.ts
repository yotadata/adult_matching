'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

export type AnalysisSummary = {
  total_likes: number;
  total_nope: number;
  like_ratio: number | null;
  first_decision_at: string | null;
  latest_decision_at: string | null;
  window_days: number | null;
  sample_size: number;
};

export type AnalysisTag = {
  tag_id: string;
  tag_name: string;
  tag_group_name: string | null;
  likes: number;
  nopes: number;
  like_ratio: number | null;
  share: number | null;
  last_liked_at: string | null;
  representative_video: {
    id: string;
    title: string | null;
    thumbnail_url: string | null;
    thumbnail_vertical_url?: string | null;
    product_url: string | null;
  } | null;
};

export type AnalysisPerformer = {
  performer_id: string;
  performer_name: string;
  likes: number;
  nopes: number;
  like_ratio: number | null;
  share: number | null;
  last_liked_at: string | null;
  representative_video: {
    id: string;
    title: string | null;
    thumbnail_url: string | null;
    thumbnail_vertical_url?: string | null;
    product_url: string | null;
  } | null;
};

export type AnalysisDecision = {
  video_id: string;
  title: string | null;
  decision_type: 'like' | 'nope';
  decided_at: string;
  thumbnail_url: string | null;
  thumbnail_vertical_url?: string | null;
  product_url: string | null;
  tags: Array<{ id: string; name: string; tag_group_name?: string | null }>;
  performers: Array<{ id: string; name: string }>;
};

export type AnalysisResultsResponse = {
  summary: AnalysisSummary;
  top_tags: AnalysisTag[];
  top_performers: AnalysisPerformer[];
  recent_decisions: AnalysisDecision[];
};

type UseAnalysisResultsOptions = {
  windowDays?: number | null;
  includeNope?: boolean;
  tagLimit?: number;
  performerLimit?: number;
  recentLimit?: number;
  enabled?: boolean;
};

export function useAnalysisResults({
  windowDays,
  includeNope,
  tagLimit,
  performerLimit,
  recentLimit,
  enabled = true,
}: UseAnalysisResultsOptions) {
  const [data, setData] = useState<AnalysisResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      setData(null);
      return;
    }
    let cancelled = false;
    const fetchAnalysis = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data: session } = await supabase.auth.getSession();
        const token = session?.session?.access_token;
        const headers: HeadersInit = {};
        if (token) headers.Authorization = `Bearer ${token}`;

        const body: Record<string, unknown> = {};
        if (windowDays !== undefined) body.window_days = windowDays;
        if (includeNope !== undefined) body.include_nope = includeNope;
        if (tagLimit !== undefined) body.tag_limit = tagLimit;
        if (performerLimit !== undefined) body.performer_limit = performerLimit;
        if (recentLimit !== undefined) body.recent_limit = recentLimit;

        const { data: payload, error } = await supabase.functions.invoke<AnalysisResultsResponse>('analysis-results', {
          headers,
          body,
        });
        if (error) throw error;
        if (!cancelled) setData(payload);
      } catch (e: unknown) {
        if (cancelled) return;
        const message = e instanceof Error ? e.message : 'あなたの性癖の取得に失敗しました';
        setError(message);
        setData(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchAnalysis();
    return () => {
      cancelled = true;
    };
  }, [enabled, windowDays, includeNope, tagLimit, performerLimit, recentLimit]);

  return { data, loading, error };
}
