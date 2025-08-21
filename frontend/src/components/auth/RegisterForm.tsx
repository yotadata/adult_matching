'use client';

import Input from './Input';
import { useForm, SubmitHandler } from 'react-hook-form';
import { useState } from 'react';
import toast from 'react-hot-toast';

type RegisterFormInputs = {
  email: string;
  password: string;
  confirmPassword: string;
};

interface RegisterFormProps {
  onClose: () => void;
}

const RegisterForm: React.FC<RegisterFormProps> = ({ onClose }) => {
  const { register, handleSubmit, watch, formState: { errors } } = useForm<RegisterFormInputs>();
  const password = watch('password');
  const [message, setMessage] = useState<{ type: 'error'; text: string } | null>(null);
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
        body: JSON.stringify({ email: data.email, password: data.password }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || '登録に失敗しました');
      }

      toast.success('確認メールを送信しました。メールボックスをご確認ください。');
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
        className="w-full py-3 px-4 bg-gradient-to-r from-purple-400 to-amber-400 hover:from-purple-500 hover:to-amber-500 text-white font-bold rounded-lg transition-all duration-300 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={isLoading}
      >
        {isLoading ? '登録中...' : '利用規約に同意して登録'}
      </button>
    </form>
  );
};

export default RegisterForm;
