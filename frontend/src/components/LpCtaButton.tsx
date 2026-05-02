'use client';

import Link from 'next/link';
import { ChevronRight } from 'lucide-react';
import { trackEvent } from '@/lib/analytics';

type Props = {
  href: string;
  label: string;
  eventName: string;
  className?: string;
  style?: React.CSSProperties;
  iconSize?: number;
};

export default function LpCtaButton({ href, label, eventName, className, style, iconSize = 18 }: Props) {
  return (
    <Link
      href={href}
      className={className}
      style={style}
      onClick={() => trackEvent(eventName)}
    >
      {label}
      <ChevronRight size={iconSize} />
    </Link>
  );
}
