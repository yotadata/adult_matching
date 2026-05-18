'use client';

import Link from 'next/link';
import { trackEvent } from '@/lib/analytics';

type Props = {
  href: string;
  videoId: string;
  source?: string;
};

export default function FanzaLinkButton({ href, videoId, source = 'video_page' }: Props) {
  return (
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={() => trackEvent('fanza_link_click', { video_id: videoId, source })}
      className="block w-full text-center bg-amber-500 hover:bg-amber-400 text-white font-bold rounded-lg py-3 text-sm transition-colors"
    >
      FANZAで見る
    </Link>
  );
}
