'use client';

import Link from 'next/link';
import { personalityTypes } from '@/data/personalityQuiz';
import { motion } from 'framer-motion';

export default function TypesPage() {
  const types = Object.entries(personalityTypes);

  return (
    <div className="p-4 md:p-8">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold text-white">16の性癖タイプ</h1>
        <p className="text-white/80 mt-2">あなたや他の人のタイプについてもっと知りましょう。</p>
      </motion.div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {types.map(([typeKey, typeData], index) => (
          <Link href={`/personality/result/${typeKey}`} key={typeKey}>
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="h-full flex flex-col justify-center items-center p-6 bg-black/20 rounded-xl border border-white/20 hover:bg-black/40 hover:border-white/30 transition-all duration-200 cursor-pointer shadow-lg"
            >
              <h3 className="text-xl font-bold text-amber-400 text-center">{typeData.title}</h3>
              <p className="text-sm text-white/70 mt-2 text-center">{typeKey}</p>
            </motion.div>
          </Link>
        ))}
      </div>
    </div>
  );
}
