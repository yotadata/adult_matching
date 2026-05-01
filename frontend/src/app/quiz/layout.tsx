import type { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
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
    <div className="min-h-screen flex flex-col" style={{ background: '#fff8f0' }}>
      <QuizHeader />
      <div className="flex-1">{children}</div>
      <footer className="flex flex-col items-center py-8 gap-3 border-t border-[#e8c9a0]/40">
        <Link href="/">
          <Image src="/seiheki_lab_header.png" alt="性癖ラボ" width={160} height={44} className="object-contain opacity-80" />
        </Link>
        <p className="text-[11px] text-[#b5541a]/50 font-bold">© 性癖ラボ</p>
      </footer>
    </div>
  );
}
