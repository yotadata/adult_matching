'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { useState } from 'react';
import toast from 'react-hot-toast';
import { buildPseudoEmail, normalizeUsername } from '@/lib/authUtils';
import TermsModal from './TermsModal';
import { trackEvent } from '@/lib/analytics';

type RegisterFormInputs = {
  username: string;
  displayName: string;
  password: string;
  confirmPassword: string;
  isAdult: boolean;
};

interface RegisterFormProps {
  onClose: () => void;
}

const RegisterForm: React.FC<RegisterFormProps> = ({ onClose }) => {
  const { register, handleSubmit, watch, formState: { errors } } = useForm<RegisterFormInputs>({
    defaultValues: { isAdult: false },
  });
  const password = watch('password');
  const [message, setMessage] = useState<{ type: 'error'; text: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTermsOpen, setIsTermsOpen] = useState(false);

  const onSubmit: SubmitHandler<RegisterFormInputs> = async (data) => {
    setMessage(null);
    setIsLoading(true);
    try {
      const { supabase } = await import('@/lib/supabase');
      const normalizedUsername = normalizeUsername(data.username);
      const pseudoEmail = buildPseudoEmail(normalizedUsername);
      const displayName = data.displayName.trim() || normalizedUsername;
      const { data: signUpData, error } = await supabase.auth.signUp({
        email: pseudoEmail,
        password: data.password,
        options: {
          data: {
            display_name: displayName,
            username: normalizedUsername,
          },
        },
      });
      if (error) throw new Error(error.message || '登録に失敗しました');

      trackEvent('sign_up', { method: 'password' });
      if (signUpData.session) {
        // autoConfirm 環境では即ログイン状態
        await supabase.auth.getUser();
        toast.success('登録してログインしました！');
      } else {
        toast.success('登録しました！');
      }
      onClose();
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
      <div className="space-y-1">
        <Input
          id="username"
          label="ユーザーID"
          type="text"
          placeholder="例: neko123"
          {...register('username', {
            required: 'ユーザーIDは必須です',
            minLength: { value: 3, message: '3文字以上で入力してください' },
            pattern: {
              value: /^[a-zA-Z0-9_]+$/,
              message: '英数字とアンダーバーのみ利用できます',
            },
          })}
        />
        {errors.username && <p className="text-red-500 text-xs">{errors.username.message}</p>}
        <p className="text-[11px] text-gray-500">※3文字以上、英数字とアンダーバーのみ／小文字に置き換えて管理します</p>
      </div>

      <Input
        id="displayName"
        label="表示名"
        type="text"
        placeholder="例: ねこ好きさん"
        {...register('displayName', {
          required: '表示名は必須です',
          maxLength: { value: 30, message: '30文字以内で入力してください' },
        })}
      />
      {errors.displayName && <p className="text-red-500 text-xs mt-1">{errors.displayName.message}</p>}

      <div className="space-y-2">
        <Input
          id="password"
          label="パスワード"
          type="password"
          placeholder="8文字以上"
          {...register('password', {
            required: 'パスワードは必須です',
            minLength: { value: 8, message: 'パスワードは8文字以上です' },
          })}
        />
        {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password.message}</p>}

        <Input
          id="confirmPassword"
          label=""
          type="password"
          placeholder="パスワード（確認用）"
          {...register('confirmPassword', {
            required: '確認用パスワードは必須です',
            validate: (value) => value === password || 'パスワードが一致しません',
          })}
        />
        {errors.confirmPassword && <p className="text-red-500 text-xs mt-1">{errors.confirmPassword.message}</p>}
      </div>

      <div className="space-y-2 text-sm text-gray-700">
        <label className="flex items-start gap-2">
          <input
            type="checkbox"
            className="mt-1 h-4 w-4 rounded border-gray-300 text-rose-500 focus:ring-rose-400"
            {...register('isAdult', { required: '18歳以上のみ登録できます' })}
          />
          <span>私は18歳以上です。</span>
        </label>
        {errors.isAdult && <p className="text-red-500 text-xs">{errors.isAdult.message}</p>}
        <p className="text-xs text-gray-500">
          利用規約は{' '}
          <button type="button" onClick={() => setIsTermsOpen(true)} className="text-rose-500 underline">
            こちら
          </button>
          {' '}から確認できます。
        </p>
      </div>

      <button
        type="submit"
        className="w-full py-3 px-4 bg-transparent border border-[#FF6B81] text-[#FF6B81] hover:bg-[#FF6B81] hover:text-white font-bold rounded-lg transition-all duration-300 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={isLoading}
      >
        {isLoading ? '登録中...' : '利用規約に同意して登録'}
      </button>

      <TermsModal open={isTermsOpen} onClose={() => setIsTermsOpen(false)} />
    </form>
  );
};

export default RegisterForm;
