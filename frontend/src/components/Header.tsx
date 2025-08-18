import Image from 'next/image'; // Imageコンポーネントをインポート

const Header = () => (
  <header className="w-full max-w-md mt-4 mb-2">
    <div className="flex justify-between items-center text-white">
      {/* <h1>adult_matching</h1> の代わりに画像を表示 */}
      <Image
        src="/seiheki_lab.png" // publicディレクトリからのパス
        alt="Seiheki Lab Logo"
        width={230} // 画像の幅を大きく設定
        height={100} // 画像の高さを大きく設定
        priority // LCP改善のため、優先的に読み込む
        draggable="false" // 画像を選択できないようにする
        style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }} // ロゴに影をつける
      />
      <button className="p-4 py-2 mx-2 text-sm font-bold text-white rounded-xl border border-white/50 hover:bg-white/20 transition-colors duration-200">
        ログイン
      </button>
    </div>
  </header>
);

export default Header;
