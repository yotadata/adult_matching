'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { QUESTIONS, calcType } from './data';

type Gender = 'male' | 'female' | 'other';

const TOTAL_STEPS = QUESTIONS.length + 1; // 設問8 + 性別1

export default function QuizPage() {
  const router = useRouter();
  const [step, setStep] = useState(0); // 0〜7: 設問, 8: 性別
  const [answers, setAnswers] = useState<Record<number, 'a' | 'b'>>({});
  const [animating, setAnimating] = useState(false);

  const progress = ((step) / TOTAL_STEPS) * 100;
  const isGenderStep = step === QUESTIONS.length;
  const currentQ = QUESTIONS[step];

  const handleAnswer = (value: 'a' | 'b') => {
    if (animating) return;
    setAnimating(true);
    const newAnswers = { ...answers, [currentQ.id]: value };
    setAnswers(newAnswers);
    setTimeout(() => {
      setStep((s) => s + 1);
      setAnimating(false);
    }, 250);
  };

  const handleGender = (gender: Gender) => {
    if (animating) return;
    setAnimating(true);
    const type = calcType(answers);
    setTimeout(() => {
      router.push(`/quiz/result/${type}?gender=${gender}`);
    }, 300);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12"
      style={{ background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 50%, #ffd1dc 100%)' }}>

      {/* ヘッダー */}
      <div className="w-full max-w-sm mb-8 text-center">
        <p className="text-xs font-bold tracking-[0.3em] text-[#b5541a]/70 uppercase mb-1">性癖パーソナリティ診断</p>
        <h1 className="text-2xl font-black text-[#5c2e00]">あなたのタイプは？</h1>
      </div>

      {/* プログレスバー */}
      <div className="w-full max-w-sm mb-8">
        <div className="flex justify-between text-xs text-[#b5541a]/60 mb-1.5">
          <span>Q{Math.min(step + 1, TOTAL_STEPS)}/{TOTAL_STEPS}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="h-3 rounded-full bg-white/40 overflow-hidden shadow-inner">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #ff6b6b, #ffd93d)',
            }}
          />
        </div>
      </div>

      {/* カード */}
      <div
        className="w-full max-w-sm transition-opacity duration-200"
        style={{ opacity: animating ? 0 : 1 }}
      >
        {!isGenderStep ? (
          /* 設問カード */
          <div className="felt-card rounded-3xl p-7 mb-6">
            <p className="text-[11px] font-bold tracking-widest text-[#b5541a]/50 uppercase mb-3">
              Question {step + 1}
            </p>
            <p className="text-xl font-black text-[#3d1a00] leading-snug mb-8">
              {currentQ.text}
            </p>
            <div className="space-y-3">
              <ChoiceButton label={currentQ.choiceA.label} onClick={() => handleAnswer('a')} colorClass="btn-a" />
              <ChoiceButton label={currentQ.choiceB.label} onClick={() => handleAnswer('b')} colorClass="btn-b" />
            </div>
          </div>
        ) : (
          /* 性別カード */
          <div className="felt-card rounded-3xl p-7 mb-6">
            <p className="text-[11px] font-bold tracking-widest text-[#b5541a]/50 uppercase mb-3">
              Last Question
            </p>
            <p className="text-xl font-black text-[#3d1a00] leading-snug mb-8">
              最後に、あなたの性別を教えてください🙌
            </p>
            <div className="space-y-3">
              <GenderButton label="男性" emoji="🙋‍♂️" onClick={() => handleGender('male')} />
              <GenderButton label="女性" emoji="🙋‍♀️" onClick={() => handleGender('female')} />
              <GenderButton label="その他・答えたくない" emoji="✨" onClick={() => handleGender('other')} />
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .felt-card {
          background: #fff8f0;
          box-shadow:
            0 2px 0 #e8c9a0,
            0 4px 0 #d4a574,
            0 6px 0 #c4956a,
            0 8px 20px rgba(100, 50, 0, 0.15);
        }
        .btn-a {
          background: #fff0e6;
          border: 2.5px solid #ffb347;
          box-shadow: 0 3px 0 #e08020;
        }
        .btn-b {
          background: #f0f4ff;
          border: 2.5px solid #7eb8f7;
          box-shadow: 0 3px 0 #4a90d9;
        }
      `}</style>
    </div>
  );
}

function ChoiceButton({ label, onClick, colorClass }: { label: string; onClick: () => void; colorClass: string }) {
  return (
    <button
      onClick={onClick}
      className={`w-full rounded-2xl px-5 py-4 text-left text-[15px] font-bold text-[#3d1a00] leading-snug active:translate-y-[2px] active:shadow-none transition-transform ${colorClass}`}
    >
      {label}
    </button>
  );
}

function GenderButton({ label, emoji, onClick }: { label: string; emoji: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full rounded-2xl px-5 py-4 text-left text-[15px] font-bold text-[#3d1a00] leading-snug active:translate-y-[2px] transition-transform"
      style={{
        background: '#fff0e6',
        border: '2.5px solid #ffb347',
        boxShadow: '0 3px 0 #e08020',
      }}
    >
      {emoji} {label}
    </button>
  );
}
