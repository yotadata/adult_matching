'use client';

import { useState } from 'react';
import { Link2, Check } from 'lucide-react';

export default function CopyLinkButton({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium border border-[#30363d] text-[#8b949e] hover:border-violet-500/50 hover:text-[#e6edf3] transition-all"
    >
      {copied ? (
        <><Check size={14} className="text-green-400" />コピーしました</>
      ) : (
        <><Link2 size={14} />リンクをコピー</>
      )}
    </button>
  );
}
