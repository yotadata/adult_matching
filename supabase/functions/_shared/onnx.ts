import * as ort from "https://esm.sh/onnxruntime-web@1.20.0";
import { getArtifactBaseUrl, getBinaryArtifact } from "../_shared/artifacts.ts";

let userSessionPromise: Promise<ort.InferenceSession> | undefined;

function configureWasmPaths() {
  const base = getArtifactBaseUrl();
  ort.env.wasm.wasmPaths = {
    "ort-wasm-simd.wasm": `${base}/ort-wasm-simd.wasm`,
    "ort-wasm.wasm": `${base}/ort-wasm.wasm`,
    "ort-wasm-threaded.wasm": `${base}/ort-wasm-threaded.wasm`,
  };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
}

async function createUserSession(): Promise<ort.InferenceSession> {
  configureWasmPaths();
  const modelData = await getBinaryArtifact("user_tower.onnx");
  return ort.InferenceSession.create(modelData, {
    executionProviders: ["wasm"],
  });
}

export async function getUserEmbeddingSession(): Promise<ort.InferenceSession> {
  if (!userSessionPromise) {
    userSessionPromise = createUserSession();
  }
  return userSessionPromise;
}

export async function inferUserEmbedding(input: Float32Array, inputName: string): Promise<Float32Array> {
  const session = await getUserEmbeddingSession();
  const tensor = new ort.Tensor("float32", input, [1, input.length]);
  const outputs = await session.run({ [inputName]: tensor });
  const embedding = outputs["user_embedding"];
  if (!embedding) {
    throw new Error("ONNX model did not return user_embedding output");
  }
  return new Float32Array(embedding.data as Float32Array);
}

export { ort };
