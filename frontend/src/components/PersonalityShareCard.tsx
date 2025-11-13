import { personalityTypes } from '@/data/personalityQuiz';

interface PersonalityShareCardProps {
  type: string;
}

export default function PersonalityShareCard({ type }: PersonalityShareCardProps) {
  const result = personalityTypes[type];

  if (!result) {
    return null;
  }

  return (
    <div
      id="personality-share-card"
      className="w-[600px] h-[315px] bg-gradient-to-br from-gray-800 to-gray-900 text-white p-8 flex flex-col justify-between"
    >
      <div>
        <p className="text-lg text-amber-400 font-bold">あなたの性癖タイプは...</p>
        <h1 className="text-5xl font-bold my-2 text-amber-300">
          {result.title}
        </h1>
      </div>
      <div className="p-4 bg-black/20 rounded-lg shadow-inner">
        <p className="text-base text-left text-white/90 line-clamp-3">
          {result.description}
        </p>
      </div>
      <div className="text-right text-sm text-white/50">
        seiheki.me
      </div>
    </div>
  );
}
