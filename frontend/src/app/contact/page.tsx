import React from 'react';
import { lightPastelGradient } from '@/constants/backgrounds';

export default function ContactPage() {
  const defaultUrl = 'https://forms.gle/LCRizZm4XKob7vEL8';
  const formUrl = process.env.NEXT_PUBLIC_GOOGLE_FORM_URL || defaultUrl;

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-white" style={{ background: lightPastelGradient }}>
      <section className="w-full max-w-4xl mx-auto rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8">
        <div className="rounded-2xl bg-white/95 text-gray-900 shadow-lg border border-white/60">
          <div className="p-8 space-y-4">
          <p className="text-xs uppercase text-gray-500 tracking-[0.4em]">Contact</p>
          <h1 className="text-2xl font-bold text-gray-900">お問い合わせ</h1>
          <p className="text-sm text-gray-600">不具合報告やご要望などは下記フォームからお気軽にお送りください。</p>
        </div>
        {formUrl ? (
          <div className="px-4 pb-8">
            <div className="relative w-full pt-[140%]">
              <iframe
                src={formUrl}
                title="Contact Form"
                className="absolute inset-0 w-full h-full rounded-xl border border-gray-200"
                loading="lazy"
                referrerPolicy="no-referrer"
              />
            </div>
          </div>
        ) : (
          <div className="px-8 pb-8 text-sm text-red-600">
            フォームURLが設定されていません。NEXT_PUBLIC_GOOGLE_FORM_URL を設定してください。
          </div>
        )}
        </div>
      </section>
    </main>
  );
}
