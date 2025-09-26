'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { quizQuestions } from '@/data/personalityQuiz';
import { calculatePersonalityType, Answer } from '@/lib/personalityCalculator';
import { motion } from 'framer-motion';

const likertScales = [
  { value: 1, text: '強く反対' },
  { value: 2, text: '反対' },
  { value: 3, text: 'やや反対' },
  { value: 4, text: 'やや賛成' },
  { value: 5, text: '賛成' },
  { value: 6, text: '強く賛成' },
];

export default function PersonalityTestPage() {
  const router = useRouter();
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [isFinished, setIsFinished] = useState(false);
  const questionRefs = useRef<(HTMLDivElement | null)[]>([]);

  const answeredQuestionIds = new Set(answers.map(a => a.questionId));

  const handleAnswer = (questionId: number, value: number) => {
    const newAnswers = answers.filter(a => a.questionId !== questionId);
    setAnswers([...newAnswers, { questionId, value }]);

    const nextQuestionIndex = quizQuestions.findIndex(q => !answeredQuestionIds.has(q.id) && q.id !== questionId);
    if (nextQuestionIndex !== -1 && questionRefs.current[nextQuestionIndex]) {
      questionRefs.current[nextQuestionIndex]?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  };
  
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

  if (isFinished) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <p className="text-white text-xl">診断結果を計算しています...</p>
        <div className="w-24 h-24 mt-4 rounded-full border-4 border-gray-500 border-t-white animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 p-4 md:p-8">
      {quizQuestions.map((question, index) => (
        <motion.div
          key={question.id}
          ref={el => questionRefs.current[index] = el}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
          className="border border-white/20 rounded-xl p-6"
        >
          <h3 className="text-xl font-bold text-white mb-4">{index + 1}. {question.text}</h3>
          <div className="flex flex-col space-y-2">
            {likertScales.map(scale => (
              <button
                key={scale.value}
                onClick={() => handleAnswer(question.id, scale.value)}
                className={`
                  w-full text-left p-3 rounded-lg transition-colors duration-200
                  ${answers.find(a => a.questionId === question.id)?.value === scale.value
                    ? 'bg-amber-500/80 text-white font-bold'
                    : 'bg-black/20 hover:bg-black/40 text-white/80'}
                `}
              >
                {scale.text}
              </button>
            ))}
          </div>
        </motion.div>
      ))}
    </div>
  );
}