'use client';

import { ListChecks, Share2, BadgeJapaneseYen, ChevronRight } from 'lucide-react';
import { trackEvent } from '@/lib/analytics';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  name: '性癖ラボ',
  url: 'https://www.seihekilab.com',
  description: '好きな作品や女優をリストにまとめて公開・シェアできるアダルト動画キュレーションサービス。',
  potentialAction: {
    '@type': 'SearchAction',
    target: 'https://www.seihekilab.com/tags',
    'query-input': 'required name=search_term_string',
  },
};

const FEATURES = [
  {
    icon: ListChecks,
    title: 'リストにまとめる',
    body: '好きな作品・女優を自分だけのリストに整理できます。テーマ別に何本でも作れます。',
    color: 'from-purple-500/20 to-violet-500/20',
    iconColor: 'text-purple-400',
  },
  {
    icon: Share2,
    title: '公開してシェアする',
    body: 'リストはURLひとつで公開・シェア可能。自分のこだわりを誰でも見られる形で残せます。',
    color: 'from-pink-500/20 to-rose-500/20',
    iconColor: 'text-pink-400',
  },
  {
    icon: BadgeJapaneseYen,
    title: '収益化できる',
    body: 'リスト経由で作品が購入されるとアフィリエイト報酬が発生します。紹介が収益につながります。',
    color: 'from-emerald-500/20 to-teal-500/20',
    iconColor: 'text-emerald-400',
  },
];

function CreateListButton({ eventName, className, style }: { eventName: string; className: string; style: React.CSSProperties }) {
  const router = useRouter();

  const handleClick = async () => {
    trackEvent(eventName);
    const { data: { user } } = await supabase.auth.getUser();
    if (user) {
      const username = user.user_metadata?.username as string | undefined;
      if (username) {
        router.push(`/u/${username}`);
      }
    } else {
      window.dispatchEvent(new CustomEvent('open-register-modal'));
    }
  };

  return (
    <button onClick={handleClick} className={className} style={style}>
      リストを作る（無料）
      <ChevronRight size={18} />
    </button>
  );
}

export default function HomeClient() {
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <main className="w-full min-h-screen text-white" style={{ background: 'linear-gradient(135deg, #1a0d2e 0%, #160d25 33%, #2a1020 66%, #1e0d1a 100%)' }}>

        {/* ヒーロー */}
        <section className="relative overflow-hidden px-4 pt-20 pb-16 text-center">
          <div className="absolute inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse 80% 50% at 50% 0%, rgba(124,58,237,0.25) 0%, transparent 70%)' }} />
          <div className="relative max-w-2xl mx-auto">
            <p className="inline-block text-xs font-bold tracking-[0.3em] uppercase text-purple-400 bg-purple-400/10 border border-purple-400/20 rounded-full px-4 py-1 mb-6">
              AV キュレーション
            </p>
            <h1 className="text-4xl sm:text-6xl font-black leading-tight mb-6">
              好きな作品・女優を<br />
              <span style={{ background: 'linear-gradient(90deg, #a78bfa, #f472b6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                リストにして紹介
              </span>
              しよう
            </h1>
            <p className="text-base sm:text-lg text-white/70 leading-relaxed mb-10">
              お気に入りをまとめて、公開して、シェアできる。<br className="hidden sm:block" />
              あなたのこだわりを形にする場所。
            </p>
            <CreateListButton
              eventName="home_hero_cta_click"
              className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-full font-black text-white text-base hover:opacity-90 transition-opacity"
              style={{ background: 'linear-gradient(90deg, #7c3aed, #ec4899)', boxShadow: '0 4px 24px rgba(124,58,237,0.4)' }}
            />
            <p className="mt-4 text-xs text-white/40">無料・18歳以上対象</p>
          </div>
        </section>

        {/* 3つの特徴 */}
        <section className="px-4 pb-16">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-xl sm:text-2xl font-black text-center mb-8">
              性癖ラボで<span className="text-purple-400">できること</span>
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

        {/* フッター CTA */}
        <section className="px-4 pb-20">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="text-2xl sm:text-3xl font-black mb-4">
              あなたのこだわり、<br />リストにしてみませんか？
            </h2>
            <p className="text-white/60 text-sm mb-8">無料・今すぐ作成できます</p>
            <CreateListButton
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
