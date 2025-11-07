// Legacy copy of the personality quiz page (previously served at /personality).
// Kept for potential future restoration; not exported through the App Router.

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { quizQuestions } from '@/data/personalityQuiz';
import { calculatePersonalityType, Answer } from '@/lib/personalityCalculator';
import LikertScale from '@/components/LikertScale';

export default function PersonalityTestLegacy() {
  const router = useRouter();
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [isFinished, setIsFinished] = useState(false);

  useEffect(() => {
    if (answers.length === quizQuestions.length) {
      setIsFinished(true);
      const finalType = calculatePersonalityType(answers);
      localStorage.setItem('personalityResultType', finalType);
      setTimeout(() => {
        router.push(`/personality/result/${finalType}`);
      }, 1000);
    }
  }, [answers, router]);

  const handleAnswer = (questionId: number, value: number) => {
    const newAnswers = answers.filter((answer) => answer.questionId !== questionId);
    setAnswers([...newAnswers, { questionId, value }]);
  };

  if (isFinished) {
    return (
      <div className="flex flex-col items-center justify-center h-[50vh]">
        <p className="text-gray-800 text-xl">診断結果を計算しています...</p>
        <div className="w-16 h-16 mt-6 rounded-full border-4 border-gray-400 border-t-gray-800 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-12 p-4 py-10 sm:p-8 md:p-12">
      {quizQuestions.map((question, index) => (
        <div key={question.id} className="text-center">
          <h3 className="text-lg font-bold text-gray-800 mb-5">
            <span className="text-gray-800/60 mr-2">{index + 1}.</span>
            {question.text}
          </h3>
          <LikertScale
            value={answers.find((answer) => answer.questionId === question.id)?.value}
            onChange={(value) => handleAnswer(question.id, value)}
          />
        </div>
      ))}
    </div>
  );
}
