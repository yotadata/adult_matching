import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'スワイプで好みを学習 | 性癖ラボ',
  description: 'スワイプするだけでAIがあなたの性癖を学習します。登録不要・無料で今すぐ利用できます。',
  alternates: { canonical: 'https://www.seihekilab.com/swipe' },
  openGraph: {
    url: 'https://www.seihekilab.com/swipe',
  },
};

export default function SwipeLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
