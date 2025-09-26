'use client';

import Link from 'next/link';

export default function PersonalityPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-white">
      <h1 className="text-4xl font-bold mb-8">性癖パーソナリティ診断</h1>
      <p className="mb-8">診断コンテンツはここに表示されます。</p>
      <Link href="/" className="px-4 py-2 rounded-md bg-white/20 hover:bg-white/30 backdrop-blur border border-white/40">
        ホームに戻る
      </Link>
    </div>
  );
}