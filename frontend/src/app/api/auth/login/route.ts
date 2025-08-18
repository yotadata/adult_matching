import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function POST(request: Request) {
  const { userId, password } = await request.json();

  // Supabaseのauth.signInWithPasswordはメールアドレスが必須のため、signupと同様にuserIdをメールアドレスとして扱う
  const { data, error } = await supabase.auth.signInWithPassword({
    email: `${userId}@example.com`, // signup時と同じダミーのメールアドレス形式
    password: password,
  });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 400 });
  }

  return NextResponse.json({ message: 'Logged in successfully', user: data.user });
}
