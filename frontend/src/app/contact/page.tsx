import React from 'react';

export default function ContactPage() {
  const defaultUrl = 'https://forms.gle/LCRizZm4XKob7vEL8';
  const formUrl = process.env.NEXT_PUBLIC_GOOGLE_FORM_URL || defaultUrl;

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-white">
      <section className="w-full max-w-4xl mx-auto rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8 text-white space-y-6">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.35em] text-white/60">Contact</p>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">お問い合わせ</h1>
          <p className="text-sm text-white/80">不具合報告やご要望などは下記フォームからお気軽にお送りください。</p>
        </header>

        <div className="rounded-2xl bg-white/95 text-gray-900 shadow-lg border border-white/60 p-6">
          {formUrl ? (
            <div className="relative w-full pt-[140%]">
              <iframe
                src={formUrl}
                title="Contact Form"
                className="absolute inset-0 w-full h-full rounded-xl border border-gray-200"
                loading="lazy"
                referrerPolicy="no-referrer"
              />
            </div>
          ) : (
            <div className="text-sm text-red-600">
              フォームURLが設定されていません。NEXT_PUBLIC_GOOGLE_FORM_URL を設定してください。
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
