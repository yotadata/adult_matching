import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.43.0";
import { buildUserFeatureVector } from "./feature-builder.ts";
import { inferUserEmbedding } from "./onnx.ts";
import { fetchUserFeatureInput } from "./user-data.ts";

interface ComputeOptions {
  persist?: boolean;
}

export async function computeUserEmbedding(
  client: SupabaseClient,
  userId: string,
  options: ComputeOptions = {},
) {
  const { persist = true } = options;
  const context = await fetchUserFeatureInput(client, userId);
  const featureVector = await buildUserFeatureVector(context);
  const embeddingArray = await inferUserEmbedding(
    featureVector.vector,
    featureVector.artifacts.schema.input_name,
  );
  const embedding = Array.from(embeddingArray);

  if (persist) {
    const { error } = await client
      .from("user_embeddings")
      .upsert({
        user_id: userId,
        embedding,
      });
    if (error) {
      throw new Error(`Failed to upsert user_embeddings: ${error.message}`);
    }
  }

  return {
    embedding,
    summary: featureVector.summary,
  };
}
