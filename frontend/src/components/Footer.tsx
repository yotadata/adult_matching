import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="border-t border-[#21262d] mt-16 py-10 px-4 text-sm text-[#656d76]">
      <div className="max-w-5xl mx-auto">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 mb-8">
          <div>
            <p className="font-semibold text-[#8b949e] mb-3">コンテンツ</p>
            <ul className="space-y-2">
              <li><Link href="/grid" className="hover:text-[#e6edf3] transition-colors">おすすめ動画</Link></li>
              <li><Link href="/swipe" className="hover:text-[#e6edf3] transition-colors">スワイプ</Link></li>
            </ul>
          </div>
          <div>
            <p className="font-semibold text-[#8b949e] mb-3">カテゴリ</p>
            <ul className="space-y-2">
              <li><Link href="/tags" className="hover:text-[#e6edf3] transition-colors">タグ一覧</Link></li>
              <li><Link href="/performers" className="hover:text-[#e6edf3] transition-colors">女優一覧</Link></li>
            </ul>
          </div>
          <div>
            <p className="font-semibold text-[#8b949e] mb-3">診断</p>
            <ul className="space-y-2">
              <li><Link href="/quiz" className="hover:text-[#e6edf3] transition-colors">偏愛16診断</Link></li>
            </ul>
          </div>
        </div>
        <p className="text-xs text-center text-[#484f58]">© 2025 性癖ラボ. 18歳以上対象.</p>
      </div>
    </footer>
  );
}
