'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { useState } from 'react';
import { useRouter } from 'next/navigation'; // useRouterをインポート

type LoginFormInputs = {
  userId: string;
  password: string;
};

const LoginForm = () => {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormInputs>();
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter(); // useRouterを初期化

  const onSubmit: SubmitHandler<LoginFormInputs> = async (data) => {
    setMessage(null);
    setIsLoading(true);
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userId: data.userId, password: data.password }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'ログインに失敗しました');
      }

      setMessage({ type: 'success', text: 'ログイン成功！' });
      // ログイン成功後、トップページにリダイレクト
      router.push('/');
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || '予期せぬエラーが発生しました' });
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
        id="userId"
        label="ユーザーID"
        type="text"
        placeholder="ユーザーIDを入力"
        {...register('userId', {
          required: 'ユーザーIDは必須です',
        })}
      />
      {errors.userId && <p className="text-red-500 text-xs mt-1">{errors.userId.message}</p>}

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