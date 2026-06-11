'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { Loader2 } from 'lucide-react';

/** /my/lists は廃止。ログインユーザーの /u/[username] にリダイレクト */
export default function MyListsRedirectPage() {
  const router = useRouter();

  useEffect(() => {
    supabase.auth.getUser().then(async ({ data: { user } }) => {
      if (!user) { router.replace('/'); return; }
      const username = user.user_metadata?.username as string | undefined;
      if (username) {
        router.replace(`/u/${username}`);
      } else {
        router.replace('/');
      }
    });
  }, [router]);

  return (
    <div className="min-h-screen bg-[#0d1117] flex items-center justify-center">
      <Loader2 className="text-violet-400 animate-spin" size={24} />
    </div>
  );
}
