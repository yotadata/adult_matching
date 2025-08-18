import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function POST(request: Request) {
  const { email, password } = await request.json(); // userId から email に変更

  const { data, error } = await supabase.auth.signInWithPassword({
    email: email, // email を直接使用
    password: password,
  });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 400 });
  }

  return NextResponse.json({ message: 'Logged in successfully', user: data.user });
}