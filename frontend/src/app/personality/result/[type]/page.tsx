'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { personalityTypes } from '@/data/personalityQuiz';
import { useState, useRef } from 'react';
import { toPng } from 'html-to-image';
import ShareModal from '@/components/ShareModal';
import PersonalityShareCard from '@/components/PersonalityShareCard';
import { Share2 } from 'lucide-react';

export default function ResultPage() {
  const params = useParams();
  const type = params.type as string;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const result = personalityTypes[type];

  const handleShare = async () => {
    if (!cardRef.current) {
      return;
    }

    try {
      const dataUrl = await toPng(cardRef.current, { cacheBust: true });
      setImageUrl(dataUrl);
      setIsModalOpen(true);
    } catch (err) {
      console.error('oops, something went wrong!', err);
    }
  };

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen text-white">
        <h1 className="text-2xl font-bold mb-4">診断結果が見つかりません</h1>
        <p className="mb-8">URLが正しいか確認してください。</p>
        <Link href="/swipe" className="px-6 py-2 rounded-full bg-white text-gray-900 font-bold">
          スワイプに戻る
        </Link>
      </div>
    );
  }

  const shareUrl = typeof window !== 'undefined' ? window.location.href : '';
  const shareText = `私の性癖タイプは「${result.title}」でした！ あなたも診断してみよう！ #性癖診断`;

  return (
    <>
      <div className="flex flex-col items-center justify-center min-h-screen text-white px-4 text-center">
        <div className="max-w-2xl">
          <p className="text-lg text-amber-400 font-bold">あなたの性癖タイプは...</p>
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold my-4 text-amber-300">
            {result.title}
          </h1>
          <div className="p-6 mt-4 bg-black/20 rounded-lg shadow-inner">
            <p className="text-base sm:text-lg md:text-xl text-left text-white/90">
              {result.description}
            </p>
          </div>

          <div className="mt-10 flex flex-col sm:flex-row gap-4">
            <Link
              href="/swipe"
              className="w-full sm:w-auto px-8 py-3 rounded-full bg-white text-gray-900 font-bold shadow-lg hover:bg-gray-200 transition-all duration-300 transform hover:scale-105"
            >
              スワイプに戻る
            </Link>
            <button
              onClick={handleShare}
              className="w-full sm:w-auto px-8 py-3 rounded-full bg-blue-500 text-white font-bold shadow-lg hover:bg-blue-600 transition-all duration-300 transform hover:scale-105 flex items-center justify-center gap-2"
            >
              <Share2 size={20} />
              結果をシェア
            </button>
          </div>
        </div>
      </div>
      <ShareModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        imageUrl={imageUrl}
        shareUrl={shareUrl}
        shareText={shareText}
      />
      {/* Hidden card for capturing */}
      <div ref={cardRef} className="absolute -left-[9999px] top-0">
        <PersonalityShareCard type={type} />
      </div>
    </>
  );
}
