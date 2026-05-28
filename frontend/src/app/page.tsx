import type { Metadata } from 'next';
import { Brain, Zap, LayoutGrid, FlaskConical } from 'lucide-react';
import LpCtaButton from '@/components/LpCtaButton';

export const metadata: Metadata = {
  title: '性癖ラボ | AIがあなたの好みを学ぶアダルト動画サービス',
  description: 'スワイプするほどAIがあなたの性癖を学習し、好みにぴったりのアダルト動画を提案します。登録不要・無料で今すぐ利用できます。',
  robots: { index: false, follow: true },
  alternates: { canonical: 'https://www.seihekilab.com' },
  openGraph: {
    title: '性癖ラボ | AIがあなたの好みを学ぶアダルト動画サービス',
    description: 'スワイプするほどAIがあなたの性癖を学習し、好みにぴったりのアダルト動画を提案します。',
    url: 'https://www.seihekilab.com',
  },
};

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  name: '性癖ラボ',
  url: 'https://www.seihekilab.com',
  description: 'スワイプするほどAIがあなたの性癖を学習し、好みにぴったりのアダルト動画を提案するサービス。',
  potentialAction: {
    '@type': 'SearchAction',
    target: 'https://www.seihekilab.com/tags',
    'query-input': 'required name=search_term_string',
  },
};

const FEATURES = [
  {
    icon: Zap,
    title: 'スワイプで直感操作',
    body: '「気になる」か「スキップ」の2択。登録不要でその場から始められます。',
    color: 'from-yellow-500/20 to-orange-500/20',
    iconColor: 'text-yellow-400',
  },
  {
    icon: Brain,
    title: 'AIが好みを自動学習',
    body: '20枚ほど判断するとAIがあなた専用のレコメンドを組み立てます。見るたびに精度が上がります。',
    color: 'from-purple-500/20 to-violet-500/20',
    iconColor: 'text-purple-400',
  },
  {
    icon: LayoutGrid,
    title: 'あなただけのグリッド',
    body: '好みの作品が並ぶグリッド画面。気になる作品はリストに保存していつでも見返せます。',
    color: 'from-pink-500/20 to-rose-500/20',
    iconColor: 'text-pink-400',
  },
];

export default function HomePage() {
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <main className="w-full min-h-screen text-white" style={{ background: 'linear-gradient(135deg, #1a0d2e 0%, #160d25 33%, #2a1020 66%, #1e0d1a 100%)' }}>

        {/* ヒーロー */}
        <section className="relative overflow-hidden px-4 pt-20 pb-16 text-center">
          <div className="absolute inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse 80% 50% at 50% 0%, rgba(124,58,237,0.25) 0%, transparent 70%)' }} />
          <div className="relative max-w-2xl mx-auto">
            <p className="inline-block text-xs font-bold tracking-[0.3em] uppercase text-purple-400 bg-purple-400/10 border border-purple-400/20 rounded-full px-4 py-1 mb-6">
              AI × アダルト動画
            </p>
            <h1 className="text-4xl sm:text-6xl font-black leading-tight mb-6">
              あなたの性癖を<br />
              <span style={{ background: 'linear-gradient(90deg, #a78bfa, #f472b6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                AIが学ぶ
              </span>
            </h1>
            <p className="text-base sm:text-lg text-white/70 leading-relaxed mb-10">
              スワイプするだけ。見るたびに精度が上がる、<br className="hidden sm:block" />
              あなた専用のアダルト動画レコメンドサービス。
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <LpCtaButton
                href="/grid"
                label="おすすめ動画を見る"
                eventName="home_hero_grid_cta_click"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-full font-black text-white text-base hover:opacity-90 transition-opacity"
                style={{ background: 'linear-gradient(90deg, #7c3aed, #ec4899)', boxShadow: '0 4px 24px rgba(124,58,237,0.4)' }}
              />
              <LpCtaButton
                href="/about"
                label="詳しく見る"
                eventName="home_hero_about_cta_click"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-full font-black text-white/80 text-base border border-white/20 hover:bg-white/5 transition-colors"
              />
            </div>
            <p className="mt-4 text-xs text-white/40">登録不要・無料・18歳以上対象</p>
          </div>
        </section>

        {/* 3つの特徴 */}
        <section className="px-4 pb-16">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-xl sm:text-2xl font-black text-center mb-8">
              性癖ラボの<span className="text-purple-400">3つの特徴</span>
            </h2>
            <div className="grid sm:grid-cols-3 gap-4">
              {FEATURES.map((f) => {
                const Icon = f.icon;
                return (
                  <div key={f.title} className={`rounded-2xl border border-white/10 bg-gradient-to-br ${f.color} p-5 backdrop-blur-sm`}>
                    <div className={`inline-flex p-2 rounded-xl bg-white/10 mb-3 ${f.iconColor}`}>
                      <Icon size={20} />
                    </div>
                    <p className="font-bold text-white text-[15px] mb-1">{f.title}</p>
                    <p className="text-sm text-white/60 leading-relaxed">{f.body}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* 診断バナー */}
        <section className="px-4 pb-16">
          <div className="max-w-2xl mx-auto">
            <div className="rounded-3xl border border-orange-400/30 p-8 text-center relative overflow-hidden"
              style={{ background: 'linear-gradient(135deg, rgba(255,140,0,0.15), rgba(255,80,80,0.1))' }}>
              <div className="absolute inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse 60% 60% at 50% 50%, rgba(255,140,0,0.1) 0%, transparent 70%)' }} />
              <div className="relative">
                <FlaskConical size={28} className="text-orange-400 mx-auto mb-3" />
                <h2 className="text-lg font-black text-white mb-2">まず自分の性癖タイプを知る</h2>
                <p className="text-sm text-white/60 mb-5">8つの質問で、あなたの性癖が16タイプに分類されます。</p>
                <LpCtaButton
                  href="/quiz"
                  label="偏愛16診断を試す"
                  eventName="home_quiz_cta_click"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-black text-[#5c2e00] text-sm hover:opacity-90 transition-opacity"
                  style={{ background: 'linear-gradient(90deg, #ffb347, #ff8c00)' }}
                  iconSize={16}
                />
              </div>
            </div>
          </div>
        </section>

        {/* フッター CTA */}
        <section className="px-4 pb-20">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="text-2xl sm:text-3xl font-black mb-4">
              あなたの性癖、<br />AIに学ばせてみませんか？
            </h2>
            <p className="text-white/60 text-sm mb-8">登録不要・無料・今すぐ開始</p>
            <LpCtaButton
              href="/grid"
              label="今すぐ無料で始める"
              eventName="home_footer_cta_click"
              className="inline-flex items-center gap-2 px-10 py-4 rounded-full font-black text-white text-base hover:opacity-90 transition-opacity"
              style={{ background: 'linear-gradient(90deg, #7c3aed, #ec4899)', boxShadow: '0 4px 32px rgba(124,58,237,0.5)' }}
            />
          </div>
        </section>

      </main>
    </>
  );
}
