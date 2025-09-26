'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { quizQuestions } from '@/data/personalityQuiz';
import { calculatePersonalityType, Answer } from '@/lib/personalityCalculator';

const likertScales = [
  { value: 1, text: '強く反対' },
  { value: 2, text: '反対' },
  { value: 3, text: 'やや反対' },
  { value: 4, text: 'やや賛成' },
  { value: 5, text: '賛成' },
  { value: 6, text: '強く賛成' },
];

export default function QuestionsPage() {
  const router = useRouter();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [isFinished, setIsFinished] = useState(false);

  const currentQuestion = quizQuestions[currentQuestionIndex];
  const progress = ((currentQuestionIndex + 1) / quizQuestions.length) * 100;

  const handleAnswer = (value: number) => {
    const newAnswers = [...answers, { questionId: currentQuestion.id, value }];
    setAnswers(newAnswers);

    if (currentQuestionIndex < quizQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      // 全問回答完了
      setIsFinished(true);
      const finalType = calculatePersonalityType(newAnswers);
      router.push(`/personality/result/${finalType}`);
    }
  };

  if (isFinished) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen text-white">
        <p>診断結果を計算しています...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-white px-4">
      <div className="w-full max-w-2xl">
        {/* Progress Bar */}
        <div className="w-full bg-gray-700 rounded-full h-2.5 mb-8">
          <div className="bg-amber-500 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
        </div>

        {/* Question */}
        <div className="text-center">
          <p className="text-lg sm:text-xl mb-4">質問 {currentQuestionIndex + 1}/{quizQuestions.length}</p>
          <h2 className="text-2xl sm:text-3xl font-bold leading-tight">
            {currentQuestion.text}
          </h2>
        </div>

        {/* Likert Scale Options */}
        <div className="mt-10 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          {likertScales.map(({ value, text }) => (
            <button
              key={value}
              onClick={() => handleAnswer(value)}
              className="p-4 rounded-lg bg-white/10 hover:bg-white/20 backdrop-blur-sm text-white font-semibold transition-colors duration-200"
            >
              {text}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
