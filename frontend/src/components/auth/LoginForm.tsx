'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { supabase } from '@/lib/supabase';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

type LoginFormInputs = {
  email: string;
  password: string;
};

interface LoginFormProps {
  onClose: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onClose }) => {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormInputs>();
  const [message, setMessage] = useState<{ type: 'error'; text: string } | null>(null);
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
        body: JSON.stringify({ email: data.email, password: data.password }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'ログインに失敗しました');
      }

      // Set the Supabase session on the client side
      if (result.session && result.access_token) {
        await supabase.auth.setSession({
          access_token: result.access_token,
          refresh_token: result.session.refresh_token,
        });
      }

      toast.success('ログインしました！');
      onClose();
      router.push('/');
    } catch (error: unknown) {
      let errorMessage = '予期せぬエラーが発生しました';
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (
        typeof error === 'object' &&
        error !== null &&
        'error' in error &&
        typeof (error as { error: string }).error === 'string'
      ) {
        errorMessage = (error as { error: string }).error;
      }
      setMessage({ type: 'error', text: errorMessage });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {message && (
        <div className={'p-3 rounded-lg text-sm bg-red-100 text-red-800'}>
          {message.text}
        </div>
      )}
      <Input
        id="email"
        label="メールアドレス"
        type="email"
        placeholder="your@example.com"
        {...register('email', {
          required: 'メールアドレスは必須です',
          pattern: {
            value: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/,
            message: '有効なメールアドレスを入力してください',
          },
        })}
      />
      {errors.email && <p className="text-red-500 text-xs mt-1">{errors.email.message}</p>}

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
