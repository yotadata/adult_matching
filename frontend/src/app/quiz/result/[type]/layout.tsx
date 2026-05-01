import type { Metadata } from 'next';
import { QUIZ_TYPES, QuizTypeKey } from '../../data';

export async function generateMetadata({ params }: { params: Promise<{ type: string }> }): Promise<Metadata> {
  const { type } = await params;
  const quizType = QUIZ_TYPES[type as QuizTypeKey];
  if (!quizType) return {};

  const title = `私の性癖パーソナリティは「${quizType.name}」でした！`;
  const description = `${quizType.tagline} — ${quizType.description.slice(0, 60)}…`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url: `https://seihekilab.com/quiz/result/${type}`,
      siteName: '性癖ラボ',
      type: 'website',
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
    },
  };
}

export default function ResultLayout({ children }: { children: React.ReactNode }) {
  return children;
}
