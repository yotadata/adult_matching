'use client';

import { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { QUESTIONS, calcResult } from './data';

type Gender = 'male' | 'female' | 'other';

const SCALE_LABELS = ['全然違う', 'あまり違う', 'どちらでも', 'やや当てはまる', 'まさにそう'];
const TOTAL_STEPS = QUESTIONS.length + 1;

export default function QuizPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);

  const shuffled = useMemo(() => {
    const arr = [...QUESTIONS];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }, []);

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
    setTimeout(() => {
      const scoresParam = encodeURIComponent(JSON.stringify(
        Object.fromEntries(
          Object.entries(result.scores).map(([k, v]) => [k, (v as { pct: number }).pct])
        )
      ));
      router.push(`/quiz/result/${result.typeKey}?gender=${gender}&scores=${scoresParam}`);
    }, 280);
  };

  const handleBack = () => {
    if (step === 0) return;
    setStep((s) => s - 1);
    setSelected(null);
  };

  return (
    <div
      className="min-h-[calc(100vh-56px)] flex flex-col items-center px-4 py-8"
      style={{ background: 'linear-gradient(160deg, #fff5e6 0%, #ffecd2 50%, #ffd1dc 100%)' }}
    >
      {/* プログレス */}
      <div className="w-full max-w-sm mb-6">
        <div className="flex justify-between text-xs text-[#b5541a]/60 mb-1.5 font-bold">
          <span>{isGenderStep ? '最後の質問' : `Q${step + 1} / ${QUESTIONS.length}`}</span>
          <span>{Math.round(progress)}%</span>
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
            {/* 軸タグ */}
            <div>
              <p className="text-[18px] font-black text-[#3d1a00] leading-snug">
                {currentQ.text}
              </p>
            </div>

            {/* 5段階スケール */}
            <div>
              {/* エンドラベル */}
              <div className="flex justify-between text-[11px] font-bold text-[#b5541a]/60 mb-2 px-1">
                <span>全然違う</span>
                <span>まさにそう</span>
              </div>
              {/* 5つのボタン */}
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
              {/* 選択肢ラベル */}
              {selected !== null && (
                <p className="text-center text-[12px] font-bold mt-2.5" style={{ color: `hsl(${20 + selected * 20}, 70%, 45%)` }}>
                  {SCALE_LABELS[selected - 1]}
                </p>
              )}
            </div>

            {/* 次へボタン */}
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

            {/* 戻るボタン */}
            {step > 0 && (
              <button onClick={handleBack} className="text-[12px] text-[#b5541a]/50 font-bold text-center">
                ← 前の質問に戻る
              </button>
            )}
          </div>
        ) : (
          /* 性別カード */
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

function AxisBadge({ axis }: { axis: string }) {
  const map: Record<string, { label: string; color: string }> = {
    ds: { label: '支配 ⇄ 奉仕', color: '#FF6B6B' },
    nx: { label: '日常 ⇄ 非日常', color: '#74B9FF' },
    pe: { label: '快楽 ⇄ 感情', color: '#FDCB6E' },
    hl: { label: '頻度', color: '#FF8E53' },
  };
  const m = map[axis] ?? { label: axis, color: '#ccc' };
  return (
    <span
      className="text-[10px] font-black px-2.5 py-1 rounded-full text-white"
      style={{ background: m.color }}
    >
      {m.label}
    </span>
  );
}
