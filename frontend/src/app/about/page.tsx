import type { Metadata } from 'next';
import { Sparkles, Shuffle, ListChecks, BarChart3 } from 'lucide-react';

export const metadata: Metadata = {
  title: '性癖ラボとは',
  description: '性癖ラボは、スワイプするほどAIがあなたの好みを学習するアダルト動画サービスです。登録不要で今すぐ利用できます。サービスの特徴・使い方・よくある質問を紹介します。',
  openGraph: {
    title: '性癖ラボとは | AIがあなたの性癖を学ぶアダルト動画サービス',
    description: '性癖ラボは、スワイプするほどAIがあなたの好みを学習するアダルト動画サービスです。',
    url: 'https://seihekilab.com/about',
  },
  alternates: {
    canonical: 'https://seihekilab.com/about',
  },
};

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'WebPage',
  name: '性癖ラボとは',
  description: '性癖ラボは、スワイプするほどAIがあなたの好みを学習するアダルト動画サービスです。',
  url: 'https://seihekilab.com/about',
  isPartOf: {
    '@type': 'WebSite',
    name: '性癖ラボ',
    url: 'https://seihekilab.com',
  },
  mainEntity: {
    '@type': 'FAQPage',
    mainEntity: [
      {
        '@type': 'Question',
        name: '性癖ラボは無料で使えますか？',
        acceptedAnswer: { '@type': 'Answer', text: 'はい、基本機能は無料でご利用いただけます。登録不要でスワイプを始められます。' },
      },
      {
        '@type': 'Question',
        name: 'アカウント登録は必要ですか？',
        acceptedAnswer: { '@type': 'Answer', text: 'スワイプ・AI検索は登録なしで利用できます。好みの学習を継続させたい場合はアカウント登録をおすすめします。' },
      },
      {
        '@type': 'Question',
        name: 'AIはどうやって好みを学習しますか？',
        acceptedAnswer: { '@type': 'Answer', text: 'スワイプで「気になる」「スキップ」を繰り返すことで、AIがあなたの好みのタグ・出演者・世界観を自動的に学習します。20枚以上の判断がたまるとパーソナライズが完了します。' },
      },
    ],
  },
};

const FLOW = [
  {
    title: 'ログインして興味を伝える',
    action: 'まずはログイン / 新規登録を行い、最初の数枚を「気になる」か「スキップ」で伝えてください。迷ったときは直感で大丈夫です。',
    result: '最低限の履歴がそろうとAIがあなたの傾向を把握し、以降の提案が一気にあなた好みに寄っていきます。',
    icon: Sparkles,
  },
  {
    title: 'スワイプで好みを磨く',
    action: '気になる作品が出てきたら「気になる」、そうでなければ「スキップ」。テンポよく進めれば5分ほどで20枚以上の判断がたまり、性癖レーダーが鮮明になります。',
    result: '「スキップ」は評価ではなく、気分違いの作品を飛ばすだけ。続けるほど好みの輪郭がくっきりし、似た世界観や出演者を優先表示します。',
    icon: Shuffle,
  },
  {
    title: 'AIで探すで条件指定',
    action: '今の気分や気になるタグ・出演者を検索欄で選ぶと、AIが「みんなのトレンド」「あなた専用」「新着」など複数の棚を即時に組み直します。',
    result: 'その場のムードに合わせた棚へ切り替わるため、探す時間を短縮しつつ新しい発見へたどり着けます。',
    icon: ListChecks,
  },
  {
    title: '気になるリストとあなたの性癖',
    action: '「気になる」とマークした作品は自動で「気になるリスト」に保存。あなたの性癖ページでは、よく選ぶタグや出演者・傾向をカード形式で確認できます。',
    result: '見返したい作品を逃さず管理でき、自分の好み変化も俯瞰できます。',
    icon: BarChart3,
  },
];

const FEATURES = [
  { title: 'スワイプで直感的に評価', body: '「気になる」か「スキップ」の2択。難しい操作は一切なく、感覚のままに使えます。' },
  { title: 'AIが好みを自動学習', body: '20枚以上の判断がたまると、あなた専用のレコメンドエンジンが起動します。' },
  { title: '気になるリストで管理', body: '気に入った作品は自動保存。あとで見返したいときにすぐ見つかります。' },
  { title: '性癖レーダーで自己分析', body: 'よく選ぶタグ・出演者・傾向をビジュアルで確認。自分の好みが言語化されます。' },
];

const FAQS = [
  {
    q: '性癖ラボは無料で使えますか？',
    a: 'はい、基本機能は無料でご利用いただけます。登録不要でスワイプをすぐに始められます。',
  },
  {
    q: 'アカウント登録は必要ですか？',
    a: 'スワイプ・AI検索は登録なしで利用できます。好みの学習を継続させたい場合はアカウント登録をおすすめします。登録時にクレジットカード情報は不要です。',
  },
  {
    q: 'AIはどうやって好みを学習しますか？',
    a: 'スワイプで「気になる」「スキップ」を繰り返すことで、AIがあなたの好みのタグ・出演者・世界観を自動的に学習します。20枚以上の判断がたまるとパーソナライズが完了します。',
  },
  {
    q: 'スキップすると評価が下がりますか？',
    a: 'いいえ。スキップは「今の気分ではない」というシグナルとして扱われます。嫌いという評価にはなりません。',
  },
];

export default function AboutPage() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-slate-900">
        <div className="max-w-5xl mx-auto space-y-10">

          {/* ヒーロー */}
          <section className="rounded-3xl border border-slate-200 bg-white p-6 sm:p-10 shadow-[0_20px_50px_rgba(15,23,42,0.15)]">
            <p className="text-xs uppercase tracking-[0.35em] text-slate-500">About</p>
            <h1 className="mt-2 text-3xl font-extrabold leading-tight text-slate-900">
              性癖ラボとは
            </h1>
            <p className="mt-4 text-sm sm:text-base text-slate-600">
              性癖ラボは、スワイプで「気になる」か「スキップ」を重ねるほどAIがあなたの好みを学習し、
              探す時間を短縮するアダルト動画サービスです。登録不要で今すぐ利用できます。
            </p>
          </section>

          {/* 特徴 */}
          <section>
            <h2 className="text-xl font-bold text-slate-800 mb-4 px-1">サービスの特徴</h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {FEATURES.map((f) => (
                <div key={f.title} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="font-bold text-slate-900 text-[15px]">{f.title}</p>
                  <p className="mt-1 text-sm text-slate-600 leading-relaxed">{f.body}</p>
                </div>
              ))}
            </div>
          </section>

          {/* 使い方 */}
          <section>
            <h2 className="text-xl font-bold text-slate-800 mb-4 px-1">4ステップの使い方</h2>
            <div className="grid gap-6">
              {FLOW.map((step, i) => (
                <article
                  key={step.title}
                  className="rounded-2xl border border-slate-200 bg-white p-6 shadow-[0_12px_35px_rgba(15,23,42,0.12)] text-slate-900"
                >
                  <div className="flex items-center gap-3">
                    <div className="rounded-2xl bg-rose-50 p-3 text-rose-500">
                      <step.icon className="h-6 w-6" />
                    </div>
                    <h3 className="text-lg font-semibold">
                      <span className="text-rose-400 mr-1">{i + 1}.</span>{step.title}
                    </h3>
                  </div>
                  <div className="mt-4 space-y-4 text-sm text-slate-700">
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-rose-400">やること</p>
                      <p className="mt-1 leading-relaxed">{step.action}</p>
                    </div>
                    <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-400">起こること</p>
                      <p className="mt-1 leading-relaxed text-slate-800">{step.result}</p>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>

          {/* FAQ */}
          <section>
            <h2 className="text-xl font-bold text-slate-800 mb-4 px-1">よくある質問</h2>
            <div className="grid gap-4">
              {FAQS.map((faq) => (
                <div key={faq.q} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="font-bold text-slate-900 text-[14px]">Q. {faq.q}</p>
                  <p className="mt-2 text-sm text-slate-600 leading-relaxed">A. {faq.a}</p>
                </div>
              ))}
            </div>
          </section>

        </div>
      </main>
    </>
  );
}
