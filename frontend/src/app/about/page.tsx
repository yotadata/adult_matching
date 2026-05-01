import type { Metadata } from 'next';
import Link from 'next/link';
import { Sparkles, Shuffle, ListChecks, BarChart3, Zap, Brain, BookMarked, FlaskConical, ChevronRight } from 'lucide-react';

export const metadata: Metadata = {
  title: '性癖ラボとは',
  description: '性癖ラボは、スワイプするほどAIがあなたの好みを学習するアダルト動画サービスです。登録不要で今すぐ利用できます。',
  openGraph: {
    title: '性癖ラボとは | AIがあなたの性癖を学ぶアダルト動画サービス',
    description: '性癖ラボは、スワイプするほどAIがあなたの好みを学習するアダルト動画サービスです。',
    url: 'https://seihekilab.com/about',
  },
  alternates: { canonical: 'https://seihekilab.com/about' },
};

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'WebPage',
  name: '性癖ラボとは',
  description: '性癖ラボは、スワイプするほどAIがあなたの好みを学習するアダルト動画サービスです。',
  url: 'https://seihekilab.com/about',
  isPartOf: { '@type': 'WebSite', name: '性癖ラボ', url: 'https://seihekilab.com' },
  mainEntity: {
    '@type': 'FAQPage',
    mainEntity: [
      { '@type': 'Question', name: '性癖ラボは無料で使えますか？', acceptedAnswer: { '@type': 'Answer', text: 'はい、基本機能は無料でご利用いただけます。登録不要でスワイプを始められます。' } },
      { '@type': 'Question', name: 'アカウント登録は必要ですか？', acceptedAnswer: { '@type': 'Answer', text: 'スワイプ・AI検索は登録なしで利用できます。好みの学習を継続させたい場合はアカウント登録をおすすめします。' } },
      { '@type': 'Question', name: 'AIはどうやって好みを学習しますか？', acceptedAnswer: { '@type': 'Answer', text: 'スワイプで「気になる」「スキップ」を繰り返すことで、AIがあなたの好みのタグ・出演者・世界観を自動的に学習します。' } },
      { '@type': 'Question', name: 'スキップすると評価が下がりますか？', acceptedAnswer: { '@type': 'Answer', text: 'いいえ。スキップは「今の気分ではない」というシグナルとして扱われます。嫌いという評価にはなりません。' } },
    ],
  },
};

const STATS = [
  { value: '登録不要', label: 'すぐ使える' },
  { value: '無料', label: '基本機能すべて' },
  { value: 'AI学習', label: '20枚でパーソナライズ完了' },
  { value: '24時間', label: 'いつでも利用可能' },
];

const FEATURES = [
  {
    icon: Zap,
    title: 'スワイプで直感操作',
    body: '「気になる」か「スキップ」の2択。難しい操作は一切なく、感覚のままに使えます。',
    color: 'from-yellow-500/20 to-orange-500/20',
    iconColor: 'text-yellow-400',
  },
  {
    icon: Brain,
    title: 'AIが好みを自動学習',
    body: '20枚以上の判断がたまると、あなた専用のレコメンドエンジンが起動します。見るたびに精度が上がります。',
    color: 'from-purple-500/20 to-violet-500/20',
    iconColor: 'text-purple-400',
  },
  {
    icon: BookMarked,
    title: '気になるリストで保存',
    body: '気に入った作品は自動保存。あとで見返したいときにすぐ見つかります。',
    color: 'from-pink-500/20 to-rose-500/20',
    iconColor: 'text-pink-400',
  },
  {
    icon: BarChart3,
    title: '性癖レーダーで自己分析',
    body: 'よく選ぶタグ・出演者・傾向をビジュアルで確認。自分の好みが言語化されます。',
    color: 'from-cyan-500/20 to-blue-500/20',
    iconColor: 'text-cyan-400',
  },
];

const FLOW = [
  {
    icon: Sparkles,
    title: 'ログインして興味を伝える',
    action: 'まずはログイン / 新規登録を行い、最初の数枚を「気になる」か「スキップ」で伝えてください。迷ったときは直感で大丈夫です。',
    result: '最低限の履歴がそろうとAIがあなたの傾向を把握し、以降の提案が一気にあなた好みに寄っていきます。',
  },
  {
    icon: Shuffle,
    title: 'スワイプで好みを磨く',
    action: '気になる作品が出てきたら「気になる」、そうでなければ「スキップ」。テンポよく進めれば5分ほどで20枚以上の判断がたまります。',
    result: '続けるほど好みの輪郭がくっきりし、似た世界観や出演者を優先表示します。',
  },
  {
    icon: ListChecks,
    title: 'AIで探すで条件指定',
    action: '今の気分や気になるタグ・出演者を検索欄で選ぶと、AIが複数の棚を即時に組み直します。',
    result: 'その場のムードに合わせた棚へ切り替わるため、探す時間を短縮しつつ新しい発見へたどり着けます。',
  },
  {
    icon: BarChart3,
    title: '気になるリストとあなたの性癖',
    action: '「気になる」とマークした作品は自動で保存。あなたの性癖ページでは傾向をカード形式で確認できます。',
    result: '見返したい作品を逃さず管理でき、自分の好み変化も俯瞰できます。',
  },
];

const FAQS = [
  { q: '性癖ラボは無料で使えますか？', a: 'はい、基本機能は無料でご利用いただけます。登録不要でスワイプをすぐに始められます。' },
  { q: 'アカウント登録は必要ですか？', a: 'スワイプ・AI検索は登録なしで利用できます。好みの学習を継続させたい場合はアカウント登録をおすすめします。登録時にクレジットカード情報は不要です。' },
  { q: 'AIはどうやって好みを学習しますか？', a: 'スワイプで「気になる」「スキップ」を繰り返すことで、AIがあなたの好みのタグ・出演者・世界観を自動的に学習します。20枚以上の判断がたまるとパーソナライズが完了します。' },
  { q: 'スキップすると評価が下がりますか？', a: 'いいえ。スキップは「今の気分ではない」というシグナルとして扱われます。嫌いという評価にはなりません。' },
];

export default function AboutPage() {
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <main className="w-full min-h-screen text-white" style={{ background: 'linear-gradient(135deg, #1a0d2e 0%, #160d25 33%, #2a1020 66%, #1e0d1a 100%)' }}>

        {/* ヒーロー */}
        <section className="relative overflow-hidden px-4 pt-16 pb-12 text-center">
          <div className="absolute inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse 80% 50% at 50% 0%, rgba(124,58,237,0.25) 0%, transparent 70%)' }} />
          <div className="relative max-w-2xl mx-auto">
            <p className="inline-block text-xs font-bold tracking-[0.3em] uppercase text-purple-400 bg-purple-400/10 border border-purple-400/20 rounded-full px-4 py-1 mb-6">
              About 性癖ラボ
            </p>
            <h1 className="text-3xl sm:text-5xl font-black leading-tight mb-6">
              AIがあなたの<br />
              <span style={{ background: 'linear-gradient(90deg, #a78bfa, #f472b6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                性癖を学ぶ
              </span>
              、<br />
              新感覚の動画サービス
            </h1>
            <p className="text-base sm:text-lg text-white/70 leading-relaxed mb-8">
              スワイプするほど精度が上がる。<br className="sm:hidden" />
              探す時間をゼロに近づける、あなた専用のレコメンドエンジン。
            </p>
            <Link
              href="/swipe"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-black text-white text-base shadow-lg hover:opacity-90 transition-opacity"
              style={{ background: 'linear-gradient(90deg, #7c3aed, #ec4899)', boxShadow: '0 4px 24px rgba(124,58,237,0.4)' }}
            >
              今すぐ無料で試す
              <ChevronRight size={18} />
            </Link>
          </div>
        </section>

        {/* 数字 */}
        <section className="px-4 pb-12">
          <div className="max-w-2xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-3">
            {STATS.map((s) => (
              <div key={s.value} className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center backdrop-blur-sm">
                <p className="text-xl font-black text-white">{s.value}</p>
                <p className="mt-1 text-xs text-white/50">{s.label}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 機能紹介 */}
        <section className="px-4 pb-16">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-xl sm:text-2xl font-black text-center mb-8">
              性癖ラボの<span className="text-purple-400">4つの特徴</span>
            </h2>
            <div className="grid sm:grid-cols-2 gap-4">
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

        {/* 使い方 */}
        <section className="px-4 pb-16">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-xl sm:text-2xl font-black text-center mb-8">
              <span className="text-pink-400">4ステップ</span>の使い方
            </h2>
            <div className="relative">
              <div className="absolute left-6 top-8 bottom-8 w-px bg-gradient-to-b from-purple-500/50 to-pink-500/50 hidden sm:block" />
              <div className="space-y-4">
                {FLOW.map((step, i) => {
                  const Icon = step.icon;
                  return (
                    <div key={step.title} className="relative flex gap-4 rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                      <div className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-black text-sm text-white z-10"
                        style={{ background: 'linear-gradient(135deg, #7c3aed, #ec4899)' }}>
                        {i + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <Icon size={16} className="text-purple-400 shrink-0" />
                          <h3 className="font-bold text-white text-[15px]">{step.title}</h3>
                        </div>
                        <p className="text-sm text-white/60 leading-relaxed mb-2">{step.action}</p>
                        <div className="rounded-xl bg-white/5 border border-white/10 px-3 py-2">
                          <p className="text-[10px] font-bold uppercase tracking-widest text-emerald-400 mb-1">起こること</p>
                          <p className="text-sm text-white/70 leading-relaxed">{step.result}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
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
                <FlaskConical size={32} className="text-orange-400 mx-auto mb-3" />
                <h2 className="text-xl font-black text-white mb-2">まずは自分の性癖タイプを知る</h2>
                <p className="text-sm text-white/60 mb-5">8つの質問に答えるだけで、あなたの性癖パーソナリティが16タイプに分類されます。</p>
                <Link
                  href="/quiz"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-black text-[#5c2e00] text-sm hover:opacity-90 transition-opacity"
                  style={{ background: 'linear-gradient(90deg, #ffb347, #ff8c00)' }}
                >
                  性癖16タイプ診断を試す
                  <ChevronRight size={16} />
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* FAQ */}
        <section className="px-4 pb-16">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-xl sm:text-2xl font-black text-center mb-8">
              よくある<span className="text-purple-400">質問</span>
            </h2>
            <div className="space-y-3">
              {FAQS.map((faq) => (
                <div key={faq.q} className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                  <p className="font-bold text-white text-[14px] mb-2">Q. {faq.q}</p>
                  <p className="text-sm text-white/60 leading-relaxed">A. {faq.a}</p>
                </div>
              ))}
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
            <Link
              href="/swipe"
              className="inline-flex items-center gap-2 px-10 py-4 rounded-full font-black text-white text-base hover:opacity-90 transition-opacity"
              style={{ background: 'linear-gradient(90deg, #7c3aed, #ec4899)', boxShadow: '0 4px 32px rgba(124,58,237,0.5)' }}
            >
              今すぐ無料で始める
              <ChevronRight size={18} />
            </Link>
          </div>
        </section>

      </main>
    </>
  );
}
