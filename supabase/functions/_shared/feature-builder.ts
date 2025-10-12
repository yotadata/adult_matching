import { getJsonArtifact } from "../_shared/artifacts.ts";

interface FeatureSchemaSegment {
  name: string;
  size: number;
  features?: string[];
  type: string;
}

interface FeatureSchema {
  version: string;
  input_name: string;
  input_dim: number;
  embedding_dim: number;
  segments: FeatureSchemaSegment[];
}

interface NormalizerStats {
  mean: number;
  std: number;
}

interface NormalizerPayload {
  user_numeric: Record<string, NormalizerStats>;
  item_numeric: Record<string, NormalizerStats>;
}

interface VocabularyPayload {
  tokens: string[];
}

export interface UserEmbeddingFeaturePayload {
  user_id: string;
  profile?: { created_at: string | null } | null;
  decision_stats?: {
    like_count?: number | null;
    nope_count?: number | null;
    decision_count?: number | null;
    recent_like_at?: string | null;
    average_like_hour?: number | null;
  } | null;
  price_stats?: {
    mean?: number | null;
    median?: number | null;
  } | null;
  tags?: string[] | null;
  performers?: string[] | null;
  now?: Date;
}

export interface ItemEmbeddingFeaturePayload {
  video_id: string;
  price?: number | null;
  duration_seconds?: number | null;
  product_released_at?: string | null;
  distribution_started_at?: string | null;
  safety_level?: number | null;
  region_codes?: string[] | null;
  tags?: string[] | null;
  performers?: string[] | null;
}

interface FeatureArtifacts {
  schema: FeatureSchema;
  normalizer: NormalizerPayload;
  tagTokens: string[];
  actressTokens: string[];
}

let artifactsPromise: Promise<FeatureArtifacts> | undefined;

async function loadArtifacts(): Promise<FeatureArtifacts> {
  if (!artifactsPromise) {
    artifactsPromise = (async () => {
      const [schema, normalizer, tagVocab, actressVocab] = await Promise.all([
        getJsonArtifact<FeatureSchema>("feature_schema.json"),
        getJsonArtifact<NormalizerPayload>("normalizer.json"),
        getJsonArtifact<VocabularyPayload>("vocab_tag.json"),
        getJsonArtifact<VocabularyPayload>("vocab_actress.json"),
      ]);
      return {
        schema,
        normalizer,
        tagTokens: tagVocab.tokens ?? [],
        actressTokens: actressVocab.tokens ?? [],
      };
    })();
  }
  return artifactsPromise;
}

function l2Normalize(vector: Float32Array): void {
  let sumSquares = 0;
  for (const value of vector) {
    sumSquares += value * value;
  }
  const norm = Math.sqrt(sumSquares);
  if (norm > 0) {
    for (let i = 0; i < vector.length; i += 1) {
      vector[i] = vector[i] / norm;
    }
  }
}

function buildHistogram(values: string[], tokens: string[]): Float32Array {
  if (!tokens.length) return new Float32Array(0);
  const index = new Map<string, number>();
  tokens.forEach((token, idx) => index.set(token, idx));
  const vec = new Float32Array(tokens.length);
  for (const value of values) {
    const idx = index.get(value);
    if (idx !== undefined) {
      vec[idx] += 1;
    }
  }
  l2Normalize(vec);
  return vec;
}

function standardize(value: number, stats: NormalizerStats | undefined): number {
  if (!stats) return value;
  const std = stats.std === 0 ? 1 : stats.std;
  return (value - stats.mean) / std;
}

function diffDays(later: Date, earlier: Date | null): number {
  if (!earlier) return 0;
  const diffMs = later.getTime() - earlier.getTime();
  return diffMs / (1000 * 60 * 60 * 24);
}

function parseDate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

export interface UserFeatureResult {
  vector: Float32Array;
  summary: {
    likeCount: number;
    nopeCount: number;
    decisionCount: number;
    tagMatchCount: number;
    actressMatchCount: number;
  };
  artifacts: FeatureArtifacts;
}

function coerceNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function coerceArray(values: unknown): string[] {
  if (!Array.isArray(values)) return [];
  return values.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
}

export async function buildUserFeatureVector(input: UserEmbeddingFeaturePayload): Promise<UserFeatureResult> {
  const artifacts = await loadArtifacts();
  const now = input.now ?? new Date();

  const decisionStats = input.decision_stats ?? {};
  const priceStats = input.price_stats ?? {};

  const likeCount = coerceNumber(decisionStats.like_count, 0);
  const nopeCount = coerceNumber(decisionStats.nope_count, 0);
  const decisionCount = coerceNumber(decisionStats.decision_count, likeCount + nopeCount);
  const likeRatioRaw = decisionCount > 0 ? likeCount / decisionCount : 0.5;
  const likeRatio = Math.min(Math.max(likeRatioRaw, 0), 1);

  const recentLikeAt = parseDate(decisionStats.recent_like_at ?? null);
  const avgLikeHourRaw = coerceNumber(decisionStats.average_like_hour, 12);
  const avgLikeHour = Math.min(Math.max(avgLikeHourRaw, 0), 23);

  const meanPrice = coerceNumber(priceStats.mean, 0);
  const medianPrice = coerceNumber(priceStats.median, 0);

  const tags = coerceArray(input.tags);
  const performers = coerceArray(input.performers);

  const profileCreated = parseDate(input.profile?.created_at ?? null);
  const tagValues = tags;
  const actressValues = performers;

  const numericFeatureNames = artifacts.schema.segments.find((segment) => segment.name === "numeric")?.features ?? [];
  const numericStats = artifacts.normalizer.user_numeric;

  const rawNumericValues: Record<string, number> = {
    account_age_days: diffDays(now, profileCreated),
    mean_price: meanPrice,
    median_price: medianPrice,
    like_ratio: likeRatio,
    like_count: likeCount,
    nope_count: nopeCount,
    decision_count: decisionCount,
    recent_like_days: diffDays(now, recentLikeAt),
  };

  const numericVector = numericFeatureNames.map((name) => standardize(rawNumericValues[name] ?? 0, numericStats[name]));

  const hourComponent = avgLikeHour / 23;

  const tagVector = buildHistogram(tagValues, artifacts.tagTokens);
  const actressVector = buildHistogram(actressValues, artifacts.actressTokens);

  const featureVector = new Float32Array([
    ...numericVector,
    hourComponent,
    ...tagVector,
    ...actressVector,
  ]);

  if (featureVector.length !== artifacts.schema.input_dim) {
    throw new Error(`Feature length mismatch: expected ${artifacts.schema.input_dim}, got ${featureVector.length}`);
  }

  return {
    vector: featureVector,
    summary: {
      likeCount,
      nopeCount,
      decisionCount,
      tagMatchCount: tagValues.length,
      actressMatchCount: actressValues.length,
    },
    artifacts,
  };
}
