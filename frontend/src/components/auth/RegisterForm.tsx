'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { useState } from 'react';

type RegisterFormInputs = {
  userId: string;
  password: string;
  confirmPassword: string;
};

const RegisterForm = () => {
  const { register, handleSubmit, watch, formState: { errors } } = useForm<RegisterFormInputs>();
  const password = watch('password');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const onSubmit: SubmitHandler<RegisterFormInputs> = async (data) => {
    setMessage(null);
    setIsLoading(true);
    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ userId: data.userId, password: data.password }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || '登録に失敗しました');
      }

      setMessage({ type: 'success', text: '登録が完了しました！ログインしてください。' });
      // 登録成功後のリダイレクトやモーダルを閉じる処理は親コンポーネントで制御することも可能
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
        placeholder="3〜15文字の半角英数字"
        {...register('userId', {
          required: 'ユーザーIDは必須です',
          minLength: { value: 3, message: 'ユーザーIDは3文字以上です' },
          maxLength: { value: 15, message: 'ユーザーIDは15文字以下です' },
          pattern: { value: /^[a-zA-Z0-9]+$/, message: 'ユーザーIDは半角英数字のみ使用できます' },
        })}
      />
      {errors.userId && <p className="text-red-500 text-xs mt-1">{errors.userId.message}</p>}

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
        label="パスワード（確認用）"
        type="password"
        placeholder="パスワードを再入力"
        {...register('confirmPassword', {
          required: '確認用パスワードは必須です',
          validate: (value) => value === password || 'パスワードが一致しません',
        })}
      />
      {errors.confirmPassword && <p className="text-red-500 text-xs mt-1">{errors.confirmPassword.message}</p>}

      <button
        type="submit"
        className="w-full py-3 px-4 bg-amber-400 hover:bg-amber-500 text-gray-900 font-bold rounded-lg transition-colors duration-300 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={isLoading}
      >
        {isLoading ? '登録中...' : '利用規約に同意して登録'}
      </button>
    </form>
  );
};

export default RegisterForm;