'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { QUESTIONS, QUIZ_TYPES, QuizTypeKey, calcResult } from './data';
import { generateSessionId, trackEvent } from '@/lib/analytics';
import { supabase } from '@/lib/supabase';

type Gender = 'male' | 'female' | 'other';

const SCALE_LABELS = ['全然違う', 'あまり違う', 'どちらでも', 'やや当てはまる', 'まさにそう'];
const TOTAL_STEPS = QUESTIONS.length + 1;
const STORAGE_KEY_PROGRESS = 'quiz_progress';
const STORAGE_KEY_RESULT = 'quiz_result';
const STORAGE_KEY_ANONYMOUS_SESSION_ID = 'quiz_anonymous_session_id';

function getAnonymousQuizSessionId() {
  try {
    const existing = localStorage.getItem(STORAGE_KEY_ANONYMOUS_SESSION_ID);
    if (existing) return existing;
    const created = generateSessionId();
    localStorage.setItem(STORAGE_KEY_ANONYMOUS_SESSION_ID, created);
    return created;
  } catch {
    return generateSessionId();
  }
}

async function persistQuizResult(params: {
  typeKey: QuizTypeKey;
  gender: Gender;
  answers: Record<number, number>;
  scores: Record<string, number>;
  anonymousSessionId: string;
}) {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    const payload = {
      user_id: user?.id ?? null,
      anonymous_session_id: params.anonymousSessionId,
      result_type: params.typeKey,
      gender: params.gender,
      axis_scores: params.scores,
      answers: params.answers,
    };
    const { error } = await supabase.from('quiz_diagnosis_results').insert(payload);
    if (error) {
      console.error('Failed to persist quiz diagnosis result:', error.message);
    }
  } catch (error) {
    console.error('Unexpected error persisting quiz diagnosis result:', error);
  }
}

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
  const startTrackedRef = useRef(false);

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

  // quiz_start を一度だけ発火
  useEffect(() => {
    if (!ready || startTrackedRef.current) return;
    startTrackedRef.current = true;
    trackEvent('quiz_start');
  }, [ready]);

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

  const handleGender = async (gender: Gender) => {
    if (animating) return;
    setAnimating(true);
    const result = calcResult(answers);
    const anonymousSessionId = getAnonymousQuizSessionId();
    const normalizedScores = Object.fromEntries(
      Object.entries(result.scores).map(([k, v]) => [k, (v as { pct: number }).pct])
    );
    trackEvent('quiz_complete', { type: result.typeKey, gender });
    const scoresParam = encodeURIComponent(JSON.stringify(normalizedScores));
    // 結果を保存・進捗をクリア
    try {
      localStorage.setItem(STORAGE_KEY_RESULT, JSON.stringify({
        typeKey: result.typeKey,
        scores: normalizedScores,
        gender,
        completedAt: new Date().toISOString(),
      }));
      localStorage.removeItem(STORAGE_KEY_PROGRESS);
    } catch {}
    await persistQuizResult({
      typeKey: result.typeKey,
      gender,
      answers,
      scores: normalizedScores,
      anonymousSessionId,
    });
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

  const STITCH_CARD = {
    background: '#fffdf5',
    borderRadius: '24px',
    border: '2px solid #e0c090',
    outline: '2px dashed rgba(180,120,60,0.3)',
    outlineOffset: '-8px',
    boxShadow: '0 3px 0 #c8946a, 0 6px 20px rgba(100,50,0,0.10)',
  };

  return (
    <div className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8">
      {/* 前回の結果バナー */}
      {prevResult && step === 0 && (
        <div className="w-full max-w-sm mb-4 rounded-2xl px-4 py-3 flex items-center justify-between gap-2"
          style={{ background: '#fffdf5', border: '2px dashed rgba(180,120,60,0.35)', boxShadow: '0 2px 0 #c8946a' }}>
          <p className="text-[12px] text-[#7a4a1a] font-bold">前回の結果: <span className="text-[#c05a00]">{prevResult.name}</span></p>
          <Link href={`/quiz/result/${prevResult.typeKey}?gender=${prevResult.gender}&scores=${prevResult.scores}`}
            className="text-[11px] font-black px-3 py-1 rounded-full text-white shrink-0"
            style={{ background: '#c87941', boxShadow: '0 2px 0 #9e5a28' }}>
            見る
          </Link>
        </div>
      )}

      {/* プログレス */}
      <div className="w-full max-w-sm mb-6">
        <div className="flex justify-between text-xs text-[#b5541a]/60 mb-2 font-bold">
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
        {/* ドット式プログレスバー */}
        <div className="flex gap-1.5">
          {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
            <div
              key={i}
              className="flex-1 h-2 rounded-full transition-all duration-300"
              style={{
                background: i < step
                  ? '#c87941'
                  : i === step
                  ? '#e8a060'
                  : 'rgba(180,120,60,0.18)',
              }}
            />
          ))}
        </div>
      </div>

      {/* カード */}
      <div
        className="w-full max-w-sm transition-opacity duration-200 flex-1 flex flex-col"
        style={{ opacity: animating ? 0 : 1 }}
      >
        {!isGenderStep ? (
          <div className="p-6 flex flex-col gap-6" style={STITCH_CARD}>
            <p className="text-[18px] font-black text-[#3d1a00] leading-snug">
              {currentQ.text}
            </p>

            {/* 5段階スケール */}
            <div>
              <div className="flex justify-between text-[11px] font-bold text-[#b5541a]/50 mb-2.5 px-1">
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
                            background: `hsl(${20 + v * 20}, 85%, 55%)`,
                            color: '#fff',
                            boxShadow: `0 3px 0 hsl(${20 + v * 20}, 85%, 38%), inset 0 0 0 2px rgba(255,255,255,0.3)`,
                            transform: 'translateY(-2px)',
                            border: '2px solid transparent',
                          }
                        : {
                            background: '#fdf0e0',
                            color: '#c8956a',
                            border: '2px dashed rgba(180,120,60,0.3)',
                            boxShadow: '0 2px 0 #d4a574',
                          }
                    }
                  >
                    {v}
                  </button>
                ))}
              </div>
              {selected !== null && (
                <p className="text-center text-[12px] font-bold mt-2.5" style={{ color: `hsl(${20 + selected * 20}, 70%, 42%)` }}>
                  {SCALE_LABELS[selected - 1]}
                </p>
              )}
            </div>

            <button
              onClick={handleNext}
              disabled={selected === null}
              className="w-full rounded-2xl py-4 font-black text-[15px] transition-all active:translate-y-[1px]"
              style={
                selected !== null
                  ? { background: '#c87941', color: '#fff', boxShadow: '0 4px 0 #9e5a28', border: '2px solid #e8a060' }
                  : { background: '#e8d8c0', color: '#c8a880', cursor: 'not-allowed', border: '2px dashed rgba(180,120,60,0.2)' }
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
          <div className="p-6" style={STITCH_CARD}>
            <p className="text-[11px] font-black tracking-widest text-[#b5541a]/40 uppercase mb-3">✦ Last Question ✦</p>
            <p className="text-[18px] font-black text-[#3d1a00] leading-snug mb-6">
              最後に、あなたの性別を教えてください
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
                  className="w-full rounded-2xl px-5 py-4 text-left text-[15px] font-bold text-[#3d1a00] active:translate-y-[1px] transition-transform"
                  style={{ background: '#fdf0e0', border: '2px dashed rgba(180,120,60,0.4)', boxShadow: '0 3px 0 #c8946a' }}
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
