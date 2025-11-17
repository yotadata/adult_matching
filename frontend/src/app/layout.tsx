import type { Metadata } from "next";
import "./globals.css";
import ClientLayout from './ClientLayout';
import { Toaster } from 'react-hot-toast';
import Script from 'next/script'; // ★ 追加

export const metadata: Metadata = {
  title: "性癖ラボ",
  description: "アダルト動画とのマッチング",
  icons: {
    icon: [
      { url: '/seiheki_icon.png' },
      { url: '/icon.png' },
    ],
    shortcut: ['/seiheki_icon.png'],
    apple: ['/apple-icon.png'],
  },
  openGraph: {
    title: "性癖ラボ",
    description: "アダルト動画とのマッチング",
    images: ['/opengraph-image.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: "性癖ラボ",
    description: "アダルト動画とのマッチング",
    images: ['/twitter-image.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <ClientLayout>
          {children}
        </ClientLayout>
        <Toaster />
        {/* Google Analytics */} {/* ★ ここから追加 */}
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
        {/* ★ ここまで追加 */}
      </body>
    </html>
  );
}
