import { Sparkles, Shuffle, ListChecks, BarChart3 } from 'lucide-react';

const FLOW = [
  {
    title: '1. ログインして興味を伝える',
    action: 'まずはログイン / 新規登録を行い、最初の数枚を「気になる」か「スキップ」で伝えてください。迷ったときは直感で大丈夫です。',
    result: '最低限の履歴がそろうと AI があなたの傾向を把握し、以降の提案が一気にあなた好みに寄っていきます。',
    icon: Sparkles,
  },
  {
    title: '2. スワイプで好みを磨く',
    action: '気になる作品が出てきたら「気になる」、そうでなければ「スキップ」。テンポよく進めれば 5 分ほどで 20 枚以上の判断がたまり、性癖レーダーが鮮明になります。',
    result: '「スキップ」は評価ではなく、気分違いの作品を飛ばすだけ。続けるほど好みの輪郭がくっきりし、似た世界観や出演者を優先表示します。',
    icon: Shuffle,
  },
  {
    title: '3. AIで探すで条件指定',
    action: '今の気分や気になるタグ・出演者を検索欄で選ぶと、AIが「みんなのトレンド」「あなた専用」「新着」など複数の棚を即時に組み直します。',
    result: 'その場のムードに合わせた棚へ切り替わるため、探す時間を短縮しつつ新しい発見へたどり着けます。',
    icon: ListChecks,
  },
  {
    title: '4. 気になるリストとあなたの性癖',
    action: '「気になる」とマークした作品は自動で「気になるリスト」に保存。あなたの性癖ページでは、よく選ぶタグや出演者・傾向をカード形式で確認できます。',
    result: '見返したい作品を逃さず管理でき、自分の嗜好変化も俯瞰できます。推しポイントをシェアして楽しむことも可能です。',
    icon: BarChart3,
  },
];

export default function AboutPage() {
  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-slate-900">
      <div className="max-w-5xl mx-auto space-y-10">
        <section className="rounded-3xl border border-slate-200 bg-white p-6 sm:p-10 shadow-[0_20px_50px_rgba(15,23,42,0.15)]">
          <p className="text-xs uppercase tracking-[0.35em] text-slate-500">About</p>
          <h1 className="mt-2 text-3xl font-extrabold leading-tight text-slate-900">このサイトについて</h1>
          <p className="mt-4 text-sm sm:text-base text-slate-600">
            性癖ラボは、スワイプで「気になる」か「スキップ」を重ねるほど AI があなたの好みを学習し、探す時間を短縮します。
            4 つのステップでやるべきことと結果が明確になるよう設計しています。
          </p>
        </section>

        <section className="grid gap-6">
          {FLOW.map((step) => (
            <article
              key={step.title}
              className="rounded-2xl border border-slate-200 bg-white p-6 shadow-[0_12px_35px_rgba(15,23,42,0.12)] text-slate-900"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-rose-50 p-3 text-rose-500">
                  <step.icon className="h-6 w-6" />
                </div>
                <h2 className="text-lg font-semibold">{step.title}</h2>
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
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600">
          <p>
            迷ったら `/swipe` で 10 枚以上の作品にリアクションしてください。AI が即座にあなたのダッシュボードを最適化し、「AIで探す」や「あなたの性癖」での分析にも反映されます。
            気になることがあればいつでもヘッダーの「お問い合わせ」からご連絡ください。
          </p>
        </section>
      </div>
    </main>
  );
}
