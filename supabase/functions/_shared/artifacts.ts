const baseUrl = Deno.env.get("MODEL_ARTIFACT_BASE_URL");

if (!baseUrl) {
  console.warn("MODEL_ARTIFACT_BASE_URL is not set. Edge Functions that rely on Two-Tower artifacts will fail to load.");
}

const jsonCache = new Map<string, Promise<any>>();
const binaryCache = new Map<string, Promise<ArrayBuffer>>();

async function fetchJson(path: string) {
  if (!baseUrl) throw new Error("MODEL_ARTIFACT_BASE_URL is not configured");
  const url = `${baseUrl.replace(/\/$/, "")}/${path}`;
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`Failed to fetch artifact ${path}: ${response.status}`);
  }
  return response.json();
}

async function fetchBinary(path: string) {
  if (!baseUrl) throw new Error("MODEL_ARTIFACT_BASE_URL is not configured");
  const url = `${baseUrl.replace(/\/$/, "")}/${path}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch binary artifact ${path}: ${response.status}`);
  }
  return response.arrayBuffer();
}

export async function getJsonArtifact<T>(filename: string): Promise<T> {
  if (!jsonCache.has(filename)) {
    jsonCache.set(filename, fetchJson(filename));
  }
  return jsonCache.get(filename)!;
}

export async function getBinaryArtifact(filename: string): Promise<ArrayBuffer> {
  if (!binaryCache.has(filename)) {
    binaryCache.set(filename, fetchBinary(filename));
  }
  return binaryCache.get(filename)!;
}

export function getArtifactBaseUrl(): string {
  if (!baseUrl) throw new Error("MODEL_ARTIFACT_BASE_URL is not configured");
  return baseUrl.replace(/\/$/, "");
}
