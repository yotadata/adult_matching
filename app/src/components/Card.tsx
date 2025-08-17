import React from 'react';
import { motion } from 'framer-motion';
import type { Video } from '@/lib/data';

interface CardProps {
  video: Video;
}

const Card: React.FC<CardProps> = ({ video }) => {
  const youtubeEmbedUrl = `https://www.youtube.com/embed/${video.videoId}`;

  return (
    <motion.div
      className="w-full h-full bg-black/25 backdrop-blur-2xl rounded-2xl shadow-2xl overflow-hidden flex flex-col"
      style={{
        border: '1px solid rgba(255, 255, 255, 0.18)',
      }}
    >
      <div className="relative w-full h-3/5">
        <iframe
          src={youtubeEmbedUrl}
          title={video.title}
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="w-full h-full"
        ></iframe>
      </div>
      <div className="p-4 text-white flex-grow flex flex-col">
        <h2 className="text-2xl font-bold font-poppins line-clamp-2">{video.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {video.tags.map((tag) => (
            <span key={tag} className="px-2 py-1 text-xs bg-white/20 rounded-full">
              {tag}
            </span>
          ))}
        </div>
        <p className="text-sm text-gray-200 line-clamp-3 flex-grow">{video.description}</p>
      </div>
    </motion.div>
  );
};

export default Card;