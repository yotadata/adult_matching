'use client';

import Header from '@/components/Header';
import useMediaQuery from '@/hooks/useMediaQuery';
import Image from 'next/image';
import Link from 'next/link';

// ダミーデータの型定義
interface LikedVideo {
  id: number;
  title: string;
  imageUrl: string;
  price: number;
  productUrl: string;
}

// ダミーデータ
const dummyLikedVideos: LikedVideo[] = [
  {
    id: 1,
    title: '【VR】VR専用機材じゃないとダメなんでしょ？って思ってた時期が俺にもありました。',
    imageUrl: '/images/sample-thumb-1.jpg', // 仮の画像パス
    price: 2980,
    productUrl: '#',
  },
  {
    id: 2,
    title: '新人グラビアアイドル！初めての撮影で緊張…！',
    imageUrl: '/images/sample-thumb-2.jpg', // 仮の画像パス
    price: 1980,
    productUrl: '#',
  },
  {
    id: 3,
    title: '会社の美人上司と禁断の社内恋愛',
    imageUrl: '/images/sample-thumb-3.jpg', // 仮の画像パス
    price: 2480,
    productUrl: '#',
  },
];

const LikedVideosPage = () => {
  const isMobile = useMediaQuery('(max-width: 768px)'); // mdブレークポイント

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="p-4 sm:p-8">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-6">
            いいねした動画
          </h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {dummyLikedVideos.map((video) => (
              <div key={video.id} className="bg-white rounded-lg shadow-md overflow-hidden flex flex-col">
                <div className="relative w-full h-48">
                  {/* <Image src={video.imageUrl} alt={video.title} layout="fill" objectFit="cover" /> */}
                  <div className="w-full h-full bg-gray-200 flex items-center justify-center">
                    <span className="text-gray-500">画像プレースホルダー</span>
                  </div>
                </div>
                <div className="p-4 flex-grow flex flex-col">
                  <h2 className="text-md font-bold text-gray-800 flex-grow">{video.title}</h2>
                  <div className="mt-4 flex justify-between items-center">
                    <p className="text-lg font-bold text-amber-500">{`￥${video.price.toLocaleString()}`}</p>
                    <Link href={video.productUrl} passHref>
                      <button className="bg-amber-400 text-gray-900 font-bold py-2 px-4 rounded-lg hover:bg-amber-500 transition-colors">
                        購入ページへ
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default LikedVideosPage;
