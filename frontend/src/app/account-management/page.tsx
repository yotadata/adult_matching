import { lightPastelGradient } from '@/constants/backgrounds';
import { Edit3, Key, LogOut, Trash2 } from 'lucide-react';

const cards = [
  { title: '表示名の変更', icon: Edit3, description: 'プロフィールに表示される名前を変更できます。' },
  { title: 'パスワードの変更', icon: Key, description: '現在のパスワードと新しいパスワードを入力して更新します。' },
];

export default function AccountManagementPage() {
  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-white" style={{ background: lightPastelGradient }}>
      <section className="w-full max-w-4xl mx-auto rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8">
        <div className="rounded-2xl bg-white/95 text-gray-900 shadow-lg border border-white/60 p-6 sm:p-8 space-y-8">
          <header className="space-y-3">
            <p className="text-xs uppercase tracking-[0.35em] text-gray-400">Account</p>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">アカウント設定</h1>
            <p className="text-sm text-gray-600">
              AI にあなたの好みを学習してもらうにはアカウントが必要です。ログイン情報や履歴の管理はこのページから行えます。
            </p>
          </header>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {cards.map((card) => {
              const Icon = card.icon;
              return (
                <article key={card.title} className="rounded-xl border border-gray-200 bg-white/90 p-4 shadow-sm">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center">
                      <Icon size={18} />
                    </div>
                    <h2 className="text-lg font-semibold">{card.title}</h2>
                  </div>
                  <p className="mt-3 text-sm text-gray-600 leading-relaxed">{card.description}</p>
                </article>
              );
            })}
          </div>

          <div className="rounded-xl border border-gray-200 bg-white/90 p-5 space-y-4">
            <h2 className="text-lg font-semibold text-gray-900">その他のアクション</h2>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-full bg-gray-100 text-gray-500 flex items-center justify-center">
                  <LogOut size={16} />
                </div>
                <div>
                  <p className="font-semibold text-gray-900">ログアウト</p>
                  <p className="text-xs text-gray-500 mt-1">ヘッダー右上のメニューからいつでもログアウトできます。共有端末では忘れずにセッションを終了してください。</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center">
                  <Trash2 size={16} />
                </div>
                <div>
                  <p className="font-semibold text-gray-900">アカウント削除</p>
                  <p className="text-xs text-gray-500 mt-1">履歴を完全に削除したい場合は下のお問い合わせフォームからご依頼ください。数日以内に処理状況をご連絡します。</p>
                  
                </div>
              </div>
            </div>
          </div>

        </div>
      </section>
    </main>
  );
}
