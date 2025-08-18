import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function POST(request: Request) {
  const { userId, password } = await request.json();

  // Supabaseのauth.signUpはメールアドレスが必須のため、userIdをメールアドレスとして扱う
  // または、auth.signUp後にuser_metadataを更新してuserIdを保存する
  // 今回はシンプルにuserIdをemailとして扱う
  const { data, error } = await supabase.auth.signUp({
    email: `${userId}@example.com`, // ダミーのメールアドレス形式
    password: password,
    options: {
      data: {
        user_id_display: userId, // 表示用のユーザーIDをメタデータに保存
      },
    },
  });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 400 });
  }

  // ユーザーIDの一意性チェックは、Supabaseのauth.signUpがemailの重複をチェックすることで代替
  // ただし、ユーザーが入力したuserIdそのものの一意性を保証するには、別途DBテーブルで管理が必要
  // 今回は、emailとして扱ったuserId@example.comの重複でチェックされる

  return NextResponse.json({ message: 'User registered successfully', user: data.user });
}
