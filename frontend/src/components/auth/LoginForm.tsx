'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { useState } from 'react';
import { useRouter } from 'next/navigation';

type LoginFormInputs = {
  email: string; // userId から email に変更
  password: string;
};

const LoginForm = () => {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormInputs>();
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const onSubmit: SubmitHandler<LoginFormInputs> = async (data) => {
    setMessage(null);
    setIsLoading(true);
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: data.email, password: data.password }), // userId から email に変更
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'ログインに失敗しました');
      }

      setMessage({ type: 'success', text: 'ログイン成功！' });
      router.push('/');
    } catch (error: unknown) {
      let errorMessage = '予期せぬエラーが発生しました';
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'object' && error !== null && 'error' in error && typeof (error as any).error === 'string') {
        errorMessage = (error as any).error; // APIからのエラーメッセージを想定
      }
      setMessage({ type: 'error', text: errorMessage });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {message && (
        <div className={`p-3 rounded-lg text-sm ${message.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {message.text}
        </div>
      )}
      <Input
        id="email" // id を email に変更
        label="メールアドレス" // label を変更
        type="email" // type を email に変更
        placeholder="your@example.com" // placeholder を変更
        {...register('email', {
          required: 'メールアドレスは必須です',
          pattern: {
            value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/,
            message: '有効なメールアドレスを入力してください',
          },
        })}
      />
      {errors.email && <p className="text-red-500 text-xs mt-1">{errors.email.message}</p>} {/* userId から email に変更 */}

      <Input
        id="password"
        label="パスワード"
        type="password"
        placeholder="パスワードを入力"
        {...register('password', {
          required: 'パスワードは必須です',
        })}
      />
      {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password.message}</p>}

      <button
        type="submit"
        className="w-full py-3 px-4 bg-amber-400 hover:bg-amber-500 text-gray-900 font-bold rounded-lg transition-colors duration-300 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={isLoading}
      >
        {isLoading ? 'ログイン中...' : 'ログイン'}
      </button>
    </form>
  );
};

export default LoginForm;
