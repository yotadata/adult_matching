'use client';

import Input from './Input';

const RegisterForm = () => {
  return (
    <form className="space-y-6">
      <Input
        id="userId"
        label="ユーザーID"
        type="text"
        placeholder="3〜15文字の半角英数字"
      />
      <Input
        id="password"
        label="パスワード"
        type="password"
        placeholder="8文字以上"
      />
      <Input
        id="confirmPassword"
        label="パスワード（確認用）"
        type="password"
        placeholder="パスワードを再入力"
      />
      <button
        type="submit"
        className="w-full py-3 px-4 bg-amber-400 hover:bg-amber-500 text-gray-900 font-bold rounded-lg transition-colors duration-300 shadow-lg"
      >
        利用規約に同意して登録
      </button>
    </form>
  );
};

export default RegisterForm;
