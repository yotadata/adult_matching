import Image from 'next/image';
import { useState } from 'react';
import AuthModal from './auth/AuthModal';
import useMediaQuery from '@/hooks/useMediaQuery'; // useMediaQuery をインポート

const Header = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const isMobile = useMediaQuery('(max-width: 639px)'); // Tailwind CSS の sm (640px) 未満をモバイルとする

  const handleOpenModal = () => {
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  return (
    <header id="main-header" className={`w-full ${isMobile ? 'fixed top-0 left-0 right-0 z-40 p-4 bg-gradient-to-r from-[#C4C8E3] via-[#D7D1E3] to-[#F7D7E0] to-[#F8DBB9] shadow-md' : 'max-w-md mx-auto mt-4 mb-2'}`}>
      <div className="flex justify-between items-center text-white max-w-md mx-auto">
        <Image
          src="/seiheki_lab.png"
          alt="Seiheki Lab Logo"
          width={230}
          height={100}
          priority
          draggable="false"
          style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
        />
        <button
          onClick={handleOpenModal}
          className="p-4 py-2 mx-2 text-sm font-bold text-gray-900 rounded-xl border border-gray-300 bg-white shadow-lg hover:bg-gray-100 transition-colors duration-200"
          style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
        >
          ログイン
        </button>
      </div>
      <AuthModal isOpen={isModalOpen} onClose={handleCloseModal} />
    </header>
  );
};

export default Header;
