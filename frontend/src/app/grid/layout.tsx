import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'おすすめ動画グリッド | 性癖ラボ',
  description: 'AIがあなたの好みを学習して厳選したアダルト動画をグリッドで表示します。',
  alternates: { canonical: 'https://www.seihekilab.com/grid' },
  openGraph: {
    url: 'https://www.seihekilab.com/grid',
  },
};

export default function GridLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
