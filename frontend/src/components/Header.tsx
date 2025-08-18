import Image from 'next/image';
import { useState } from 'react';
import AuthModal from './auth/AuthModal';

const Header = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleOpenModal = () => {
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  return (
    <header className="w-full max-w-md mt-4 mb-2">
      <div className="flex justify-between items-center text-white">
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
          className="p-4 py-2 mx-2 text-sm font-bold text-white rounded-xl border border-white/50 bg-white/20 backdrop-blur-md shadow-lg hover:bg-white/30 transition-colors duration-200"
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
