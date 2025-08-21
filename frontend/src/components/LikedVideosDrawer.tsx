'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { X } from 'lucide-react';
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
    {
    id: 4,
    title: 'ギャルで人妻とかいうパワーワード',
    imageUrl: '/images/sample-thumb-4.jpg', // 仮の画像パス
    price: 3200,
    productUrl: '#',
  },
    {
    id: 5,
    title: '田舎で育った純朴な彼女との初体験',
    imageUrl: '/images/sample-thumb-5.jpg', // 仮の画像パス
    price: 2800,
    productUrl: '#',
  },
    {
    id: 6,
    title: 'オフィスで秘密の残業デート',
    imageUrl: '/images/sample-thumb-6.jpg', // 仮の画像パス
    price: 3500,
    productUrl: '#',
  },
    {
    id: 7,
    title: '夏休みのビーチで出会った彼女',
    imageUrl: '/images/sample-thumb-7.jpg', // 仮の画像パス
    price: 2100,
    productUrl: '#',
  },
    {
    id: 8,
    title: '家庭教師の先生との禁断の関係',
    imageUrl: '/images/sample-thumb-8.jpg', // 仮の画像パス
    price: 2900,
    productUrl: '#',
  },
    {
    id: 9,
    title: '幼馴染との再会、そして…',
    imageUrl: '/images/sample-thumb-9.jpg', // 仮の画像パス
    price: 2600,
    productUrl: '#',
  },
    {
    id: 10,
    title: '憧れの先輩と二人きりの部室',
    imageUrl: '/images/sample-thumb-10.jpg', // 仮の画像パス
    price: 3100,
    productUrl: '#',
  },
];

interface LikedVideosDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const LikedVideosDrawer: React.FC<LikedVideosDrawerProps> = ({ isOpen, onClose }) => {
  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        {/* Backdrop */}
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
        </Transition.Child>

        {/* Drawer Panel */}
        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10 sm:pl-16">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500 sm:duration-700"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-500 sm:duration-700"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-md">
                  <div className="flex h-full flex-col bg-white shadow-xl">
                    <div className="p-6">
                      <div className="flex items-start justify-between">
                        <Dialog.Title className="text-xl font-bold text-gray-900">
                          いいねした動画
                        </Dialog.Title>
                        <div className="ml-3 flex h-7 items-center">
                          <button
                            type="button"
                            className="relative rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            onClick={onClose}
                          >
                            <span className="absolute -inset-2.5" />
                            <span className="sr-only">Close panel</span>
                            <X className="h-6 w-6" aria-hidden="true" />
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    {/* List */}
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                       <div className="space-y-4">
                        {dummyLikedVideos.map((video) => (
                          <div key={video.id} className="bg-gray-50 rounded-lg shadow-sm overflow-hidden flex items-center p-3">
                            <div className="relative w-24 h-24 flex-shrink-0 rounded-md overflow-hidden">
                               {/* <Image src={video.imageUrl} alt={video.title} layout="fill" objectFit="cover" /> */}
                               <div className="w-full h-full bg-gray-200 flex items-center justify-center">
                                 <span className="text-gray-400 text-xs">画像</span>
                               </div>
                            </div>
                            <div className="pl-4 flex-grow">
                              <h3 className="text-sm font-semibold text-gray-800 line-clamp-2">{video.title}</h3>
                              <div className="mt-2 flex justify-between items-center">
                                <p className="text-md font-bold text-amber-500">{`￥${video.price.toLocaleString()}`}</p>
                                <Link href={video.productUrl} passHref>
                                  <button className="bg-transparent border border-amber-400 text-amber-400 hover:bg-amber-400 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
                                    見る
                                  </button>
                                </Link>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default LikedVideosDrawer;
