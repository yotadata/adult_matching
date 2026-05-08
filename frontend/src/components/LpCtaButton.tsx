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
  newTab?: boolean;
};

export default function LpCtaButton({ href, label, eventName, className, style, iconSize = 18, newTab }: Props) {
  return (
    <Link
      href={href}
      className={className}
      style={style}
      target={newTab ? '_blank' : undefined}
      rel={newTab ? 'noopener noreferrer' : undefined}
      onClick={() => trackEvent(eventName)}
    >
      {label}
      <ChevronRight size={iconSize} />
    </Link>
  );
}
