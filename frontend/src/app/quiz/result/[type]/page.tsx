import { Suspense } from 'react';
import { ResultContent } from './ResultContent';
import { QuizTypeKey } from '../../data';

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
