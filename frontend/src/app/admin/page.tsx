"use client";

import React, { useEffect, useMemo, useState } from "react";

type ORT = typeof import("onnxruntime-web");

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export default function AdminModelTestPage() {
  const [ort, setOrt] = useState<ORT | null>(null);
  const [session, setSession] = useState<import("onnxruntime-web").InferenceSession | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);

  const [userMap, setUserMap] = useState<Record<string, number> | null>(null);
  const [itemMap, setItemMap] = useState<Record<string, number> | null>(null);

  const [userId, setUserId] = useState("");
  const [itemId, setItemId] = useState("");
  const [score, setScore] = useState<number | null>(null);
  const [prob, setProb] = useState<number | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const mod = await import("onnxruntime-web");
        // Try setting CDN fallback for WASM binaries in case local resolution fails
        mod.env.wasm.wasmPaths = mod.env.wasm.wasmPaths || "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/";
        if (mounted) setOrt(mod);
      } catch (e: any) {
        console.error(e);
        setModelError("onnxruntime-web の読み込みに失敗しました");
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const loadModelFromFile = async (file: File) => {
    if (!ort) return;
    setModelError(null);
    setModelLoading(true);
    try {
      const buf = await file.arrayBuffer();
      const s = await ort.InferenceSession.create(buf);
      setSession(s);
    } catch (e: any) {
      console.error(e);
      setModelError(`モデルロード失敗: ${e?.message ?? e}`);
    } finally {
      setModelLoading(false);
    }
  };

  const loadModelFromUrl = async (url: string) => {
    if (!ort) return;
    setModelError(null);
    setModelLoading(true);
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const s = await ort.InferenceSession.create(buf);
      setSession(s);
    } catch (e: any) {
      console.error(e);
      setModelError(`URLからのモデル読み込み失敗: ${e?.message ?? e}`);
    } finally {
      setModelLoading(false);
    }
  };

  const onLoadUserMap = async (file: File) => {
    try {
      const txt = await file.text();
      const obj = JSON.parse(txt);
      setUserMap(obj);
    } catch (e: any) {
      alert("user_id_map.json の読み込みに失敗しました");
    }
  };

  const onLoadItemMap = async (file: File) => {
    try {
      const txt = await file.text();
      const obj = JSON.parse(txt);
      setItemMap(obj);
    } catch (e: any) {
      alert("item_id_map.json の読み込みに失敗しました");
    }
  };

  const canRun = useMemo(() => !!session && !!userMap && !!itemMap && userId && itemId, [session, userMap, itemMap, userId, itemId]);

  const runTest = async () => {
    if (!session || !userMap || !itemMap) return;
    setRunError(null);
    setScore(null);
    setProb(null);
    try {
      const uidx = userMap[String(userId)];
      const iidx = itemMap[String(itemId)];
      if (uidx === undefined) throw new Error("user_id がマッピングに存在しません");
      if (iidx === undefined) throw new Error("video_id がマッピングに存在しません");
      // int64 tensors via BigInt64Array
      const feeds: Record<string, any> = {};
      const userTensor = new (ort as any).Tensor("int64", new BigInt64Array([BigInt(uidx)]), [1]);
      const itemTensor = new (ort as any).Tensor("int64", new BigInt64Array([BigInt(iidx)]), [1]);
      feeds["user_idx"] = userTensor;
      feeds["item_idx"] = itemTensor;
      const out = await session.run(feeds);
      // Expect output named 'score'
      const arr = out[Object.keys(out)[0]].data as Float32Array | number[];
      const logit = Number((arr as any)[0]);
      setScore(logit);
      setProb(sigmoid(logit));
    } catch (e: any) {
      console.error(e);
      setRunError(e?.message ?? String(e));
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-8">
      <h1 className="text-2xl font-semibold">Admin: モデルテスト</h1>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">1) モデル読み込み（ONNX）</h2>
        <div className="flex items-center gap-2 flex-wrap">
          <input
            type="file"
            accept=".onnx"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadModelFromFile(f);
            }}
          />
          <span className="text-sm text-gray-500">または URL から:</span>
          <ModelFromUrl onLoad={loadModelFromUrl} loading={modelLoading} />
        </div>
        <StatusLine loading={modelLoading} ok={!!session} error={modelError} okText="モデル読み込み済み" />
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">2) マッピング読込（user_id_map.json / item_id_map.json）</h2>
        <div className="flex gap-4 flex-wrap">
          <label className="block">
            <span className="text-sm">user_id_map.json</span>
            <input type="file" accept=".json" onChange={(e) => e.target.files?.[0] && onLoadUserMap(e.target.files[0])} />
          </label>
          <label className="block">
            <span className="text-sm">item_id_map.json</span>
            <input type="file" accept=".json" onChange={(e) => e.target.files?.[0] && onLoadItemMap(e.target.files[0])} />
          </label>
        </div>
        <div className="text-sm text-gray-600">
          {userMap ? `ユーザー数: ${Object.keys(userMap).length}` : "ユーザー未読込"} / {itemMap ? `アイテム数: ${Object.keys(itemMap).length}` : "アイテム未読込"}
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">3) 推論テスト</h2>
        <div className="flex flex-col gap-2 max-w-xl">
          <input className="border rounded px-3 py-2" placeholder="user_id (文字列)" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <input className="border rounded px-3 py-2" placeholder="video_id (UUID)" value={itemId} onChange={(e) => setItemId(e.target.value)} />
          <button disabled={!canRun} onClick={runTest} className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50">
            推論を実行
          </button>
        </div>
        {runError && <div className="text-red-600 text-sm">{runError}</div>}
        {(score !== null || prob !== null) && (
          <div className="text-sm">
            <div>logit: {score?.toFixed(6)}</div>
            <div>sigmoid: {prob?.toFixed(6)}</div>
          </div>
        )}
      </section>
    </div>
  );
}

function StatusLine({ loading, ok, error, okText }: { loading: boolean; ok: boolean; error: string | null; okText: string }) {
  if (loading) return <div className="text-sm text-gray-600">読み込み中...</div>;
  if (error) return <div className="text-sm text-red-600">{error}</div>;
  if (ok) return <div className="text-sm text-green-700">{okText}</div>;
  return <div className="text-sm text-gray-600">未読み込み</div>;
}

function ModelFromUrl({ onLoad, loading }: { onLoad: (url: string) => void; loading: boolean }) {
  const [url, setUrl] = useState("");
  return (
    <div className="flex items-center gap-2">
      <input className="border rounded px-3 py-2 w-72" placeholder="https://.../two_tower_latest.onnx" value={url} onChange={(e) => setUrl(e.target.value)} />
      <button disabled={!url || loading} onClick={() => onLoad(url)} className="px-3 py-2 rounded bg-gray-700 text-white disabled:opacity-50">
        読み込み
      </button>
    </div>
  );
}

