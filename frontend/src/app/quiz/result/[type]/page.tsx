import { Suspense } from 'react';
import type { Metadata } from 'next';
import { ResultContent } from './ResultContent';
import { QUIZ_TYPES, QuizTypeKey } from '../../data';

const BASE_URL = 'https://www.seihekilab.com';

export async function generateMetadata({ params, searchParams }: {
  params: Promise<{ type: string }>;
  searchParams: Promise<{ scores?: string; gender?: string }>;
}): Promise<Metadata> {
  const { type } = await params;
  const sp = await searchParams;
  const quizType = QUIZ_TYPES[type as QuizTypeKey];
  if (!quizType) return {};

  const title = `私の偏愛16診断タイプは「${quizType.name}」でした！`;
  const description = `${quizType.tagline} — ${quizType.description.slice(0, 60)}…`;

  const ogImageUrl = `${BASE_URL}/quiz/${type}.png`;
  const resultParams = new URLSearchParams();
  if (sp.scores) resultParams.set('scores', sp.scores);
  if (sp.gender) resultParams.set('gender', sp.gender);
  const resultUrl = resultParams.toString()
    ? `${BASE_URL}/quiz/result/${type}?${resultParams.toString()}`
    : `${BASE_URL}/quiz/result/${type}`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      url: resultUrl,
      siteName: '性癖ラボ',
      type: 'website',
      images: [{ url: ogImageUrl, width: 1200, height: 630, alt: quizType.name }],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: [ogImageUrl],
    },
  };
}

export default async function ResultPage({ params }: { params: Promise<{ type: string }> }) {
  const { type } = await params;
  return (
    <Suspense fallback={
      <div className="min-h-[calc(100vh-56px)] flex items-center justify-center"
        style={{ background: 'linear-gradient(135deg, #ffecd2, #ffd1dc)' }} />
    }>
      <ResultContent typeKey={type as QuizTypeKey} />
    </Suspense>
  );
}
