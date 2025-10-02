import type { VideoMeta } from "./feature-builder.ts";

export interface CandidateResult {
  video: VideoMeta & {
    title?: string | null;
    description?: string | null;
    thumbnail_url?: string | null;
    product_released_at?: string | null;
    price?: number | null;
    safety_level?: number | null;
    region_codes?: string[] | null;
  };
  similarity: number;
  distance: number;
  score: number;
}

function recencyBoost(releasedAt?: string | null): number {
  if (!releasedAt) return 0;
  const dt = new Date(releasedAt);
  if (Number.isNaN(dt.getTime())) return 0;
  const diffDays = (Date.now() - dt.getTime()) / (1000 * 60 * 60 * 24);
  if (diffDays < 0) return 0.1;
  const decay = Math.exp(-diffDays / 120);
  return Math.min(0.3, decay * 0.3);
}

function priceRegularizer(price?: number | null): number {
  if (price == null) return 0;
  if (price <= 0) return 0.02;
  if (price < 800) return 0.03;
  if (price > 4000) return -0.05;
  return 0;
}

export function rerankCandidates(
  candidates: {
    video: CandidateResult["video"];
    similarity: number;
    distance: number;
  }[],
): CandidateResult[] {
  return candidates
    .map((candidate) => {
      const base = candidate.similarity;
      const recency = recencyBoost(candidate.video.product_released_at);
      const priceAdjust = priceRegularizer(candidate.video.price ?? null);
      const score = base * 0.75 + recency + priceAdjust;
      return {
        video: candidate.video,
        similarity: candidate.similarity,
        distance: candidate.distance,
        score,
      };
    })
    .sort((a, b) => b.score - a.score);
}
