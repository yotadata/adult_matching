'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { QUESTIONS, QUIZ_TYPES, QuizTypeKey, calcResult } from './data';

type Gender = 'male' | 'female' | 'other';

const SCALE_LABELS = ['全然違う', 'あまり違う', 'どちらでも', 'やや当てはまる', 'まさにそう'];
const TOTAL_STEPS = QUESTIONS.length + 1;
const STORAGE_KEY_PROGRESS = 'quiz_progress';
const STORAGE_KEY_RESULT = 'quiz_result';

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function QuizPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [shuffled, setShuffled] = useState(QUESTIONS);
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);
  const [prevResult, setPrevResult] = useState<{ typeKey: QuizTypeKey; name: string; scores: string; gender: string } | null>(null);

  // 保存済みの進捗・結果をロード
  useEffect(() => {
    try {
      const savedResult = localStorage.getItem(STORAGE_KEY_RESULT);
      if (savedResult) {
        const r = JSON.parse(savedResult);
        const type = QUIZ_TYPES[r.typeKey as QuizTypeKey];
        if (type) setPrevResult({
          typeKey: r.typeKey,
          name: type.name,
          scores: encodeURIComponent(JSON.stringify(r.scores ?? {})),
          gender: r.gender ?? 'other',
        });
      }

      const savedProgress = localStorage.getItem(STORAGE_KEY_PROGRESS);
      if (savedProgress) {
        const { order, answers: savedAnswers, step: savedStep } = JSON.parse(savedProgress);
        const ordered = (order as number[])
          .map((id) => QUESTIONS.find((q) => q.id === id))
          .filter(Boolean) as typeof QUESTIONS;
        if (ordered.length === QUESTIONS.length) {
          setShuffled(ordered);
          setAnswers(savedAnswers);
          setStep(savedStep);
          setReady(true);
          return;
        }
      }
    } catch {}
    setShuffled(shuffle(QUESTIONS));
    setReady(true);
  }, []);

  // 進捗を保存
  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(STORAGE_KEY_PROGRESS, JSON.stringify({
        order: shuffled.map((q) => q.id),
        answers,
        step,
      }));
    } catch {}
  }, [ready, shuffled, answers, step]);

  const progress = (step / TOTAL_STEPS) * 100;
  const isGenderStep = step === shuffled.length;
  const currentQ = shuffled[step];

  const handleSelect = (value: number) => {
    if (animating) return;
    setSelected(value);
  };

  const handleNext = () => {
    if (selected === null || animating || !currentQ) return;
    setAnimating(true);
    const newAnswers = { ...answers, [currentQ.id]: selected };
    setAnswers(newAnswers);
    setSelected(null);
    setTimeout(() => {
      setStep((s) => s + 1);
      setAnimating(false);
    }, 220);
  };

  const handleGender = (gender: Gender) => {
    if (animating) return;
    setAnimating(true);
    const result = calcResult(answers);
    const scoresParam = encodeURIComponent(JSON.stringify(
      Object.fromEntries(
        Object.entries(result.scores).map(([k, v]) => [k, (v as { pct: number }).pct])
      )
    ));
    // 結果を保存・進捗をクリア
    try {
      localStorage.setItem(STORAGE_KEY_RESULT, JSON.stringify({
        typeKey: result.typeKey,
        scores: Object.fromEntries(
          Object.entries(result.scores).map(([k, v]) => [k, (v as { pct: number }).pct])
        ),
        gender,
        completedAt: new Date().toISOString(),
      }));
      localStorage.removeItem(STORAGE_KEY_PROGRESS);
    } catch {}
    setTimeout(() => {
      router.push(`/quiz/result/${result.typeKey}?gender=${gender}&scores=${scoresParam}`);
    }, 280);
  };

  const handleBack = () => {
    if (step === 0) return;
    setStep((s) => s - 1);
    setSelected(null);
  };

  const handleReset = () => {
    try { localStorage.removeItem(STORAGE_KEY_PROGRESS); } catch {}
    setShuffled(shuffle(QUESTIONS));
    setAnswers({});
    setStep(0);
    setSelected(null);
  };

  if (!ready) return null;

  return (
    <div
      className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8"
      style={{ background: 'linear-gradient(160deg, #fff5e6 0%, #ffecd2 50%, #ffd1dc 100%)' }}
    >
      {/* 前回の結果バナー */}
      {prevResult && step === 0 && (
        <div className="w-full max-w-sm mb-4 rounded-2xl px-4 py-3 flex items-center justify-between gap-2"
          style={{ background: '#fff8f0', border: '1.5px solid #ffb347' }}>
          <p className="text-[12px] text-[#7a4a1a] font-bold">前回の結果: <span className="text-[#c05a00]">{prevResult.name}</span></p>
          <Link href={`/quiz/result/${prevResult.typeKey}?gender=${prevResult.gender}&scores=${prevResult.scores}`}
            className="text-[11px] font-black px-3 py-1 rounded-full text-white shrink-0"
            style={{ background: '#ffb347' }}>
            見る
          </Link>
        </div>
      )}

      {/* プログレス */}
      <div className="w-full max-w-sm mb-6">
        <div className="flex justify-between text-xs text-[#b5541a]/60 mb-1.5 font-bold">
          <span>{isGenderStep ? '最後の質問' : `Q${step + 1} / ${QUESTIONS.length}`}</span>
          <div className="flex items-center gap-3">
            <span>{Math.round(progress)}%</span>
            {step > 0 && (
              <button onClick={handleReset} className="text-[10px] text-[#b5541a]/40 underline underline-offset-2">
                最初から
              </button>
            )}
          </div>
        </div>
        <div className="h-2.5 rounded-full bg-white/50 overflow-hidden shadow-inner">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${progress}%`, background: 'linear-gradient(90deg, #ff6b6b, #ffd93d)' }}
          />
        </div>
      </div>

      {/* カード */}
      <div
        className="w-full max-w-sm transition-opacity duration-200 flex-1 flex flex-col"
        style={{ opacity: animating ? 0 : 1 }}
      >
        {!isGenderStep ? (
          <div
            className="rounded-3xl p-6 flex flex-col gap-6"
            style={{
              background: '#fffdf8',
              boxShadow: '0 2px 0 #e8c9a0, 0 5px 0 #d4a574, 0 8px 24px rgba(100,50,0,0.12)',
            }}
          >
            <div>
              <p className="text-[18px] font-black text-[#3d1a00] leading-snug">
                {currentQ.text}
              </p>
            </div>

            {/* 5段階スケール */}
            <div>
              <div className="flex justify-between text-[11px] font-bold text-[#b5541a]/60 mb-2 px-1">
                <span>全然違う</span>
                <span>まさにそう</span>
              </div>
              <div className="flex gap-2 justify-between">
                {[1, 2, 3, 4, 5].map((v) => (
                  <button
                    key={v}
                    onClick={() => handleSelect(v)}
                    className="flex-1 aspect-square rounded-2xl flex items-center justify-center font-black text-[15px] transition-all duration-150"
                    style={
                      selected === v
                        ? {
                            background: `hsl(${20 + v * 20}, 90%, 60%)`,
                            color: '#fff',
                            boxShadow: `0 3px 0 hsl(${20 + v * 20}, 90%, 40%)`,
                            transform: 'translateY(-2px)',
                          }
                        : {
                            background: '#fff0e6',
                            color: '#c8956a',
                            boxShadow: '0 2px 0 #e8c9a0',
                          }
                    }
                  >
                    {v}
                  </button>
                ))}
              </div>
              {selected !== null && (
                <p className="text-center text-[12px] font-bold mt-2.5" style={{ color: `hsl(${20 + selected * 20}, 70%, 45%)` }}>
                  {SCALE_LABELS[selected - 1]}
                </p>
              )}
            </div>

            <button
              onClick={handleNext}
              disabled={selected === null}
              className="w-full rounded-2xl py-4 font-black text-[15px] transition-all"
              style={
                selected !== null
                  ? { background: 'linear-gradient(90deg, #ff6b6b, #ffd93d)', color: '#fff', boxShadow: '0 4px 0 #e08020' }
                  : { background: '#e8c9a0', color: '#c8a080', cursor: 'not-allowed' }
              }
            >
              次の質問へ →
            </button>

            {step > 0 && (
              <button onClick={handleBack} className="text-[12px] text-[#b5541a]/50 font-bold text-center">
                ← 前の質問に戻る
              </button>
            )}
          </div>
        ) : (
          <div
            className="rounded-3xl p-6"
            style={{
              background: '#fffdf8',
              boxShadow: '0 2px 0 #e8c9a0, 0 5px 0 #d4a574, 0 8px 24px rgba(100,50,0,0.12)',
            }}
          >
            <p className="text-[11px] font-black tracking-widest text-[#b5541a]/50 uppercase mb-3">Last Question</p>
            <p className="text-[18px] font-black text-[#3d1a00] leading-snug mb-6">
              最後に、あなたの性別を教えてください🙌
            </p>
            <div className="space-y-3">
              {([
                { label: '男性', emoji: '🙋‍♂️', value: 'male' as Gender },
                { label: '女性', emoji: '🙋‍♀️', value: 'female' as Gender },
                { label: 'その他・答えたくない', emoji: '✨', value: 'other' as Gender },
              ]).map(({ label, emoji, value }) => (
                <button
                  key={value}
                  onClick={() => handleGender(value)}
                  className="w-full rounded-2xl px-5 py-4 text-left text-[15px] font-bold text-[#3d1a00] active:translate-y-[2px] transition-transform"
                  style={{ background: '#fff0e6', border: '2.5px solid #ffb347', boxShadow: '0 3px 0 #e08020' }}
                >
                  {emoji} {label}
                </button>
              ))}
            </div>
            <button onClick={handleBack} className="text-[12px] text-[#b5541a]/50 font-bold text-center w-full mt-4">
              ← 前の質問に戻る
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
