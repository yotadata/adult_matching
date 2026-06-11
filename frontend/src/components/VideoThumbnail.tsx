'use client';

import { useState } from 'react';
import Image from 'next/image';
import { resolveThumbnail } from '@/utils/thumbnail';

interface Props {
  source?: string | null;
  thumbnailUrl?: string | null;
  imageUrls?: string[] | null;
  alt?: string;
  fill?: boolean;
  className?: string;
  sizes?: string;
}

export default function VideoThumbnail({ source, thumbnailUrl, imageUrls, alt = '', fill, className, sizes }: Props) {
  const { primary, fallback } = resolveThumbnail({ source, thumbnail_url: thumbnailUrl, image_urls: imageUrls });
  const [src, setSrc] = useState(primary);

  if (!src) return null;

  return (
    <Image
      src={src}
      alt={alt}
      fill={fill}
      className={className}
      sizes={sizes}
      onError={() => {
        if (fallback && src !== fallback) setSrc(fallback);
      }}
    />
  );
}
