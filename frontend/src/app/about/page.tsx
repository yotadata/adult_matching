import { Sparkles, Shuffle, ListChecks, BarChart3 } from 'lucide-react';

const FLOW = [
  {
    title: '1. ログインして興味を伝える',
    action: 'まずはログイン / 新規登録を行い、最初の数枚を LIKE / NOPE してください。迷ったときは気軽にスワイプしながら感覚で選んでも問題ありません。',
    result: '最低限の履歴がそろうと AI があなたの傾向を把握し、以降の提案が一気にあなた好みに寄っていきます。',
    icon: Sparkles,
  },
  {
    title: '2. スワイプで好みを磨く',
    action: '気になる作品が出てきたら LIKE、違うと思ったら NOPE。テンポよく進めれば 5 分ほどで 20 枚以上の判断がたまり、性癖レーダーが鮮明になります。',
    result: '続けるほど好みの輪郭がくっきりし、似た世界観や出演者を優先表示。レコメンド精度がどんどん向上します。',
    icon: Shuffle,
  },
  {
    title: '3. AI検索で条件指定',
    action: '今の気分や気になるタグ・出演者を検索欄で選ぶと、AI が「みんなのトレンド」「あなた専用」「新着」など複数の棚を即時に組み直します。',
    result: 'その場のムードに合わせた棚へ切り替わるため、探す時間を短縮しつつ新しい発見へたどり着けます。',
    icon: ListChecks,
  },
  {
    title: '4. 気になるリストと嗜好レポート',
    action: 'LIKE した作品は自動で「気になるリスト」に保存。インサイトページでは、よく選ぶタグや出演者・傾向をカード形式で確認できます。',
    result: '見返したい作品を逃さず管理でき、自分の嗜好変化も俯瞰できます。推しポイントをシェアして楽しむことも可能です。',
    icon: BarChart3,
  },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen w-full bg-gradient-to-br from-[#0f172a] via-[#1e1b4b] to-[#312e81] px-4 py-10 text-white">
      <div className="max-w-5xl mx-auto space-y-10">
        <section className="rounded-3xl border border-white/15 bg-white/10 backdrop-blur-xl p-8 shadow-[0_20px_50px_rgba(15,23,42,0.45)]">
          <p className="text-xs uppercase tracking-[0.35em] text-white/70">About</p>
          <h1 className="mt-2 text-3xl font-extrabold leading-tight">このサイトについて</h1>
          <p className="mt-4 text-sm sm:text-base text-white/80">
            性癖ラボは、LIKE/NOPE の履歴と Supabase 上のレコメンドを組み合わせて、あなたの「今の気分」に寄り添う作品探しをお手伝いします。
            以下の4ステップに沿って進めるだけで、やるべきことと得られる結果が一目でわかります。
          </p>
        </section>

        <section className="grid gap-6">
          {FLOW.map((step) => (
            <article
              key={step.title}
              className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-lg shadow-[0_12px_35px_rgba(15,23,42,0.35)]"
            >
              <div className="flex items-center gap-3 text-white">
                <div className="rounded-2xl bg-white/15 p-3">
                  <step.icon className="h-6 w-6" />
                </div>
                <h2 className="text-lg font-semibold">{step.title}</h2>
              </div>
              <div className="mt-4 space-y-3 text-sm text-white/85">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-rose-200/80">やること</p>
                  <p className="mt-1 leading-relaxed">{step.action}</p>
                </div>
                <div className="rounded-xl border border-white/15 bg-white/5 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-200/80">起こること</p>
                  <p className="mt-1 leading-relaxed">{step.result}</p>
                </div>
              </div>
            </article>
          ))}
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/5 p-6 text-sm text-white/75">
          <p>
            迷ったら、まずは `/swipe` で 10 枚以上の作品にリアクションしてみてください。AI が即座にあなたのダッシュボードを最適化します。
            気になることがあればいつでもヘッダーの「お問い合わせ」からご連絡ください。
          </p>
        </section>
      </div>
    </main>
  );
}
