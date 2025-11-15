import Link from 'next/link';

export default function NotFoundPage() {
  return (
    <main className="min-h-screen w-full bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white flex items-center justify-center px-4 text-center">
      <div className="space-y-4 max-w-md">
        <p className="text-sm uppercase tracking-[0.3em] text-white/50">404</p>
        <h1 className="text-2xl font-bold">お探しのページが見つかりませんでした</h1>
        <p className="text-sm text-white/70">
          リンクが古いか、すでに廃止された可能性があります。トップのスワイプ画面から改めて作品を探してみてください。
        </p>
        <div className="pt-2">
          <Link
            href="/swipe"
            className="inline-flex items-center justify-center rounded-full bg-white/90 px-5 py-2 text-sm font-semibold text-gray-900 hover:bg-white transition-colors"
          >
            スワイプ画面に戻る
          </Link>
        </div>
      </div>
    </main>
  );
}
