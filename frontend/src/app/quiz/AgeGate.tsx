'use client';

import { useState, useEffect } from 'react';

const STORAGE_KEY = 'age_verified';

export default function AgeGate() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      if (!localStorage.getItem(STORAGE_KEY)) setShow(true);
    } catch {
      setShow(true);
    }
  }, []);

  if (!show) return null;

  const handleVerified = () => {
    try { localStorage.setItem(STORAGE_KEY, '1'); } catch {}
    setShow(false);
  };

  const handleDeny = () => {
    window.location.href = 'https://www.google.com';
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center px-4"
      style={{ background: 'rgba(30, 10, 10, 0.85)', backdropFilter: 'blur(8px)' }}>
      <div className="w-full max-w-sm rounded-3xl p-8 flex flex-col items-center text-center"
        style={{ background: '#fffdf8', boxShadow: '0 8px 40px rgba(0,0,0,0.4)' }}>

        <p className="text-4xl mb-4">🔞</p>
        <h2 className="text-[22px] font-black text-[#3d1a00] mb-2">年齢確認</h2>
        <p className="text-[13px] text-[#7a4a1a] leading-relaxed mb-6">
          このコンテンツは<strong className="text-[#c05a00]">18歳以上</strong>を対象としています。<br />
          あなたは18歳以上ですか？
        </p>

        <button
          onClick={handleVerified}
          className="w-full rounded-2xl py-4 font-black text-white text-[15px] mb-3"
          style={{ background: 'linear-gradient(90deg, #ff6b6b, #ffd93d)', boxShadow: '0 4px 0 #e08020' }}
        >
          18歳以上です
        </button>
        <button
          onClick={handleDeny}
          className="w-full rounded-2xl py-3 font-bold text-[13px]"
          style={{ background: '#f0e8e0', color: '#7a4a1a' }}
        >
          18歳未満です
        </button>

        <p className="text-[10px] text-[#b5541a]/40 mt-5 leading-relaxed">
          「18歳以上です」を選択することで、<br />成人向けコンテンツの閲覧に同意したものとみなします。
        </p>
      </div>
    </div>
  );
}
