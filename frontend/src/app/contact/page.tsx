import React from 'react';

export default function ContactPage() {
  const defaultUrl = 'https://forms.gle/LCRizZm4XKob7vEL8';
  const formUrl = process.env.NEXT_PUBLIC_GOOGLE_FORM_URL || defaultUrl;

  return (
    <main className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-lg border border-gray-100">
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
    </main>
  );
}
