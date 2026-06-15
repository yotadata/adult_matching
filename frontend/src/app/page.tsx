import type { Metadata } from 'next';
import HomeClient from './HomeClient';

export const metadata: Metadata = {
  title: '性癖ラボ | お気に入りのAV作品・女優をリストにして紹介しよう',
  description: '好きな作品や女優をリストにまとめて公開・シェアできるアダルト動画キュレーションサービス。無料で今すぐリストを作れます。',
  alternates: { canonical: 'https://www.seihekilab.com' },
  openGraph: {
    title: '性癖ラボ | お気に入りのAV作品・女優をリストにして紹介しよう',
    description: '好きな作品や女優をリストにまとめて公開・シェアできるアダルト動画キュレーションサービス。',
    url: 'https://www.seihekilab.com',
  },
};

export default function HomePage() {
  return <HomeClient />;
}
