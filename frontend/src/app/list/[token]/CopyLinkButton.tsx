'use client';

import { useState } from 'react';
import { Link2, Check } from 'lucide-react';

export default function CopyLinkButton({ url, variant = 'dark' }: { url: string; variant?: 'dark' | 'light' | 'glass' }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const base = 'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold transition-all shrink-0';
  const styles = {
    dark:  'border border-[#30363d] text-[#8b949e] hover:border-violet-500/50 hover:text-violet-400',
    light: 'bg-white/20 border border-white/30 text-white hover:bg-white/30 backdrop-blur',
    glass: 'bg-white/50 border border-white/70 text-violet-700 hover:bg-white/70 backdrop-blur shadow-sm',
  };

  return (
    <button
      onClick={handleCopy}
      title={copied ? 'コピーしました' : 'リンクをコピー'}
      className={`${base} ${styles[variant]}`}
    >
      {copied
        ? <><Check size={13} />コピー済み</>
        : <><Link2 size={13} />シェア</>}
    </button>
  );
}
