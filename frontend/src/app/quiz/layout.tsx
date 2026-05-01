import type { Metadata } from 'next';
import QuizHeader from './QuizHeader';

export const metadata: Metadata = {
  title: '性癖パーソナリティ診断 | 性癖ラボ',
  description: '8つの質問に答えるだけ。あなたの性癖パーソナリティタイプを診断します。',
  openGraph: {
    title: '性癖パーソナリティ診断',
    description: '8つの質問に答えるだけ。あなたのタイプがわかる！',
    url: 'https://seihekilab.com/quiz',
    siteName: '性癖ラボ',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: '性癖パーソナリティ診断',
    description: '8つの質問に答えるだけ。あなたのタイプがわかる！',
  },
};

export default function QuizLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen" style={{ background: '#fff8f0' }}>
      <QuizHeader />
      {children}
    </div>
  );
}
