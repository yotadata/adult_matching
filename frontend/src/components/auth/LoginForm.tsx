'use client';

import Input from './Input';

const LoginForm = () => {
  return (
    <form className="space-y-6">
      <Input
        id="userId"
        label="ユーザーID"
        type="text"
        placeholder="ユーザーIDを入力"
      />
      <Input
        id="password"
        label="パスワード"
        type="password"
        placeholder="パスワードを入力"
      />
      <button
        type="submit"
        className="w-full py-3 px-4 bg-amber-400 hover:bg-amber-500 text-gray-900 font-bold rounded-lg transition-colors duration-300 shadow-lg"
      >
        ログイン
      </button>
    </form>
  );
};

export default LoginForm;
