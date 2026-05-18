import type { Metadata } from "next";
import "./globals.css";
import ClientLayout from './ClientLayout';
import { Toaster } from 'react-hot-toast';
import Script from 'next/script'; // ★ 追加

export const metadata: Metadata = {
  title: { default: "性癖ラボ | AIがあなたの好みを学ぶアダルト動画サービス", template: "%s | 性癖ラボ" },
  description: "スワイプするほどAIがあなたの性癖を学習し、好みにぴったりのアダルト動画を提案します。無料・登録不要で今すぐ利用できます。",
  metadataBase: new URL('https://www.seihekilab.com'),
  icons: {
    icon: [
      { url: '/seiheki_icon.png' },
      { url: '/icon.png' },
    ],
    shortcut: ['/seiheki_icon.png'],
    apple: ['/apple-icon.png'],
  },
  openGraph: {
    title: "性癖ラボ | AIがあなたの好みを学ぶアダルト動画サービス",
    description: "スワイプするほどAIがあなたの性癖を学習し、好みにぴったりのアダルト動画を提案します。",
    url: 'https://www.seihekilab.com',
    siteName: '性癖ラボ',
    locale: 'ja_JP',
    type: 'website',
    images: ['/opengraph-image.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: "性癖ラボ | AIがあなたの好みを学ぶアダルト動画サービス",
    description: "スワイプするほどAIがあなたの性癖を学習し、好みにぴったりのアダルト動画を提案します。",
    images: ['/twitter-image.png'],
  },
  verification: {
    google: 'IDNQh6qoo-tpim03wuI8PUzB4iLQuioKba7w29EfLXU',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className="antialiased">
        <ClientLayout>
          {children}
        </ClientLayout>
        <Toaster />
        {process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID ? (
          <>
            <Script
              strategy="afterInteractive"
              src={`https://www.googletagmanager.com/gtag/js?id=${process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID}`}
            />
            <Script
              id="google-analytics-script"
              strategy="afterInteractive"
              dangerouslySetInnerHTML={{
                __html: `
                  window.dataLayer = window.dataLayer || [];
                  function gtag(){dataLayer.push(arguments);}
                  gtag('js', new Date());
                  gtag('config', '${process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID}');
                `,
              }}
            />
          </>
        ) : null}
      </body>
    </html>
  );
}
