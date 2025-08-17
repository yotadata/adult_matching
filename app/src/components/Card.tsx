import React from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import type { Video } from '@/lib/data';

interface CardProps {
  video: Video;
}

const Card: React.FC<CardProps> = ({ video }) => {
  return (
    <motion.div
      className="absolute w-[90vw] max-w-md h-[70vh] bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl overflow-hidden cursor-grab"
      style={{
        border: '1px solid rgba(255, 255, 255, 0.3)',
      }}
    >
      <div className="relative w-full h-3/5">
        <Image
          src={video.thumbnailUrl}
          alt={video.title}
          fill
          className="object-cover"
          priority
        />
      </div>
      <div className="p-4 text-white">
        <h2 className="text-2xl font-bold font-poppins">{video.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {video.tags.map((tag) => (
            <span key={tag} className="px-2 py-1 text-sm bg-white/20 rounded-full">
              {tag}
            </span>
          ))}
        </div>
        <p className="text-sm text-gray-200 line-clamp-3">{video.description}</p>
      </div>
    </motion.div>
  );
};

export default Card;