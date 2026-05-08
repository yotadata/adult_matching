import type { Metadata } from 'next';
import Image from 'next/image';
import QuizHeader from './QuizHeader';

export const metadata: Metadata = {
  title: '性癖16タイプ診断 | 性癖ラボ',
  description: '8つの質問に答えるだけ。あなたの性癖パーソナリティタイプを診断します。',
  openGraph: {
    title: '性癖16タイプ診断',
    description: '8つの質問に答えるだけ。あなたのタイプがわかる！',
    url: 'https://seihekilab.com/quiz',
    siteName: '性癖ラボ',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: '性癖16タイプ診断',
    description: '8つの質問に答えるだけ。あなたのタイプがわかる！',
  },
};

export default function QuizLayout({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="min-h-screen flex flex-col"
      style={{
        background: 'linear-gradient(160deg, #0f0a1e 0%, #1a0d2e 50%, #0d1020 100%)',
        backgroundAttachment: 'fixed',
      }}
    >
      <QuizHeader />
      <div className="flex-1">{children}</div>
      <footer className="flex flex-col items-center py-8 gap-3" style={{ borderTop: '1px solid rgba(180,150,80,0.2)' }}>
        <Image src="/seiheki_lab_header.png" alt="性癖ラボ" width={160} height={44} className="object-contain opacity-50" />
        <p className="text-[11px] font-bold" style={{ color: 'rgba(180,150,80,0.4)' }}>© 性癖ラボ</p>
      </footer>
    </div>
  );
}
