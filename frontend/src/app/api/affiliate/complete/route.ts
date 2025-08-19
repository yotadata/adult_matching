import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/supabaseAdmin';

export async function POST(req: NextRequest) {
  const { user_id, video_id } = await req.json();

  const { error } = await supabaseAdmin
    .from('likes')
    .update({ purchased: true })
    .eq('user_id', user_id)
    .eq('video_id', video_id);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ success: true });
}
