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

  const DARK_CARD = {
    background: 'rgba(28,24,18,0.85)',
    borderRadius: '24px',
    border: '1px solid rgba(180,150,80,0.35)',
    boxShadow: '0 4px 0 rgba(0,0,0,0.4), 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(180,150,80,0.15)',
    backdropFilter: 'blur(8px)',
  };

  return (
    <div className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8">
      {/* 前回の結果バナー */}
      {prevResult && step === 0 && (
        <div className="w-full max-w-sm mb-4 rounded-2xl px-4 py-3 flex items-center justify-between gap-2"
          style={{ background: 'rgba(28,24,18,0.8)', border: '1px solid rgba(180,150,80,0.3)' }}>
          <p className="text-[12px] font-bold" style={{ color: 'rgba(200,180,140,0.7)' }}>前回の結果: <span style={{ color: '#e8d5a0' }}>{prevResult.name}</span></p>
          <Link href={`/quiz/result/${prevResult.typeKey}?gender=${prevResult.gender}&scores=${prevResult.scores}`}
            className="text-[11px] font-black px-3 py-1 rounded-full shrink-0"
            style={{ background: 'rgba(180,150,80,0.25)', color: '#e8d5a0', border: '1px solid rgba(180,150,80,0.4)' }}>
            見る
          </Link>
        </div>
      )}

      {/* プログレス */}
      <div className="w-full max-w-sm mb-6">
        <div className="flex justify-between text-xs mb-2 font-bold" style={{ color: 'rgba(200,180,140,0.5)' }}>
          <span>{isGenderStep ? '最後の質問' : `Q${step + 1} / ${QUESTIONS.length}`}</span>
          <div className="flex items-center gap-3">
            <span>{Math.round(progress)}%</span>
            {step > 0 && (
              <button onClick={handleReset} className="text-[10px] underline underline-offset-2" style={{ color: 'rgba(200,180,140,0.35)' }}>
                最初から
              </button>
            )}
          </div>
        </div>
        <div className="flex gap-1.5">
          {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
            <div
              key={i}
              className="flex-1 h-1.5 rounded-full transition-all duration-300"
              style={{
                background: i < step
                  ? 'rgba(180,150,80,0.9)'
                  : i === step
                  ? 'rgba(180,150,80,0.5)'
                  : 'rgba(180,150,80,0.12)',
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
          <div className="p-6 flex flex-col gap-6" style={DARK_CARD}>
            <p className="text-[18px] font-black leading-snug" style={{ color: '#f0e6d3' }}>
              {currentQ.text}
            </p>

            <div>
              <div className="flex justify-between text-[11px] font-bold mb-2.5 px-1" style={{ color: 'rgba(200,180,140,0.5)' }}>
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
                            background: `hsl(${240 + v * 20}, 60%, 55%)`,
                            color: '#fff',
                            boxShadow: `0 3px 0 hsl(${240 + v * 20}, 60%, 35%)`,
                            transform: 'translateY(-2px)',
                            border: '1px solid rgba(255,255,255,0.2)',
                          }
                        : {
                            background: 'rgba(180,150,80,0.08)',
                            color: 'rgba(200,180,140,0.6)',
                            border: '1px solid rgba(180,150,80,0.25)',
                          }
                    }
                  >
                    {v}
                  </button>
                ))}
              </div>
              {selected !== null && (
                <p className="text-center text-[12px] font-bold mt-2.5" style={{ color: `hsl(${240 + selected * 20}, 60%, 70%)` }}>
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
                  ? { background: 'rgba(180,150,80,0.25)', color: '#e8d5a0', border: '1px solid rgba(180,150,80,0.5)', boxShadow: '0 4px 0 rgba(0,0,0,0.3)' }
                  : { background: 'rgba(180,150,80,0.06)', color: 'rgba(200,180,140,0.3)', cursor: 'not-allowed', border: '1px solid rgba(180,150,80,0.1)' }
              }
            >
              次の質問へ →
            </button>

            {step > 0 && (
              <button onClick={handleBack} className="text-[12px] font-bold text-center" style={{ color: 'rgba(200,180,140,0.4)' }}>
                ← 前の質問に戻る
              </button>
            )}
          </div>
        ) : (
          <div className="p-6" style={DARK_CARD}>
            <p className="text-[11px] font-black tracking-widest uppercase mb-3" style={{ color: 'rgba(180,150,80,0.5)' }}>✦ Last Question ✦</p>
            <p className="text-[18px] font-black leading-snug mb-6" style={{ color: '#f0e6d3' }}>
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
                  className="w-full rounded-2xl px-5 py-4 text-left text-[15px] font-bold active:translate-y-[1px] transition-transform"
                  style={{ background: 'rgba(180,150,80,0.1)', border: '1px solid rgba(180,150,80,0.3)', color: '#e8d5a0' }}
                >
                  {emoji} {label}
                </button>
              ))}
            </div>
            <button onClick={handleBack} className="text-[12px] font-bold text-center w-full mt-4" style={{ color: 'rgba(200,180,140,0.4)' }}>
              ← 前の質問に戻る
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
