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

export interface VideoMeta {
  id: string;
  tags: string[];
  performers: string[];
  price: number | null;
  productReleasedAt?: string | null;
}

export interface DecisionRecord {
  decisionType: "like" | "nope";
  createdAt: string | null;
  video: VideoMeta | null;
}

export interface UserProfileMeta {
  createdAt: string | null;
}

export interface UserFeatureInput {
  profile: UserProfileMeta | null;
  decisions: DecisionRecord[];
  now?: Date;
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

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
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

export async function buildUserFeatureVector(input: UserFeatureInput): Promise<UserFeatureResult> {
  const artifacts = await loadArtifacts();
  const now = input.now ?? new Date();

  const likes = input.decisions.filter((d) => d.decisionType === "like" && d.video);
  const nopes = input.decisions.filter((d) => d.decisionType === "nope" && d.video);
  const decisions = input.decisions.filter((d) => d.video);

  const likeVideos = likes.map((d) => d.video!).filter(Boolean);
  const likeTimestamps = likes
    .map((d) => parseDate(d.createdAt))
    .filter((dt): dt is Date => !!dt)
    .sort((a, b) => b.getTime() - a.getTime());

  const prices = likeVideos
    .map((v) => (typeof v.price === "number" ? Number(v.price) : null))
    .filter((value): value is number => value != null);

  const tagValues = likeVideos.flatMap((v) => v.tags ?? []);
  const actressValues = likeVideos.flatMap((v) => v.performers ?? []);

  const profileCreated = parseDate(input.profile?.createdAt ?? null);

  const numericFeatureNames = artifacts.schema.segments.find((segment) => segment.name === "numeric")?.features ?? [];
  const numericStats = artifacts.normalizer.user_numeric;

  const rawNumericValues: Record<string, number> = {
    account_age_days: diffDays(now, profileCreated),
    mean_price: prices.length ? prices.reduce((sum, p) => sum + p, 0) / prices.length : 0,
    median_price: prices.length ? median(prices) : 0,
    like_ratio: decisions.length ? likes.length / decisions.length : 0.5,
    like_count: likes.length,
    nope_count: nopes.length,
    decision_count: decisions.length,
    recent_like_days: likeTimestamps.length ? diffDays(now, likeTimestamps[0]) : 0,
  };

  const numericVector = numericFeatureNames.map((name) => standardize(rawNumericValues[name] ?? 0, numericStats[name]));

  const likeHours = likeTimestamps.map((dt) => dt.getHours());
  const avgHour = likeHours.length ? likeHours.reduce((sum, h) => sum + h, 0) / likeHours.length : 12;
  const hourComponent = avgHour / 23;

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
      likeCount: likes.length,
      nopeCount: nopes.length,
      decisionCount: decisions.length,
      tagMatchCount: tagValues.length,
      actressMatchCount: actressValues.length,
    },
    artifacts,
  };
}
