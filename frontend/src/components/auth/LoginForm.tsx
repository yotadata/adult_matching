'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { supabase } from '@/lib/supabase';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import { buildPseudoEmail, normalizeUsername } from '@/lib/authUtils';
import { trackEvent } from '@/lib/analytics';

type LoginFormInputs = {
  username: string;
  password: string;
};

interface LoginFormProps {
  onClose: () => void;
}

const CTA_GRADIENT = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';
const CTA_GRADIENT_CLASS = 'from-[#ADB4E3] via-[#F7BECE] to-[#F9B1C4]';

const LoginForm: React.FC<LoginFormProps> = ({ onClose }) => {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormInputs>();
  const [message, setMessage] = useState<{ type: 'error'; text: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const onSubmit: SubmitHandler<LoginFormInputs> = async (data) => {
    setMessage(null);
    setIsLoading(true);
    try {
      // クライアントで直接ログインし、ブラウザにセッションを保存する
      const pseudoEmail = buildPseudoEmail(normalizeUsername(data.username));
      const { error } = await supabase.auth.signInWithPassword({
        email: pseudoEmail,
        password: data.password,
      });
      if (error) throw new Error(error.message || 'ログインに失敗しました');
      trackEvent('login', { method: 'password' });
      // 念のため取得して確定させる（CORS/環境差異の切り分け用）
      await supabase.auth.getUser();

      toast.success('ログインしました！');
      onClose();
      router.push('/swipe');
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
        id="username"
        label="ユーザーID"
        type="text"
        placeholder="登録したユーザーID"
        {...register('username', {
          required: 'ユーザーIDは必須です',
        })}
      />
      {errors.username && <p className="text-red-500 text-xs mt-1">{errors.username.message}</p>}

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

      <div className={`rounded-full bg-gradient-to-r ${CTA_GRADIENT_CLASS} p-[2px]`}>
        <button
          type="submit"
          className="w-full py-3 px-4 font-bold rounded-full transition-all duration-300 shadow-sm hover:shadow-md disabled:opacity-60 disabled:cursor-not-allowed bg-white hover:bg-gradient-to-r hover:from-[#ADB4E3]/15 hover:via-[#F7BECE]/15 hover:to-[#F9B1C4]/15"
          disabled={isLoading}
        >
          <span className={`bg-gradient-to-r ${CTA_GRADIENT_CLASS} bg-clip-text text-transparent`}>
            {isLoading ? 'ログイン中...' : 'ログイン'}
          </span>
        </button>
      </div>
    </form>
  );
};

export default LoginForm;
