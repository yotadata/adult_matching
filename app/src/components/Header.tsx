import React from 'react';

const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 bg-white/80 backdrop-blur-sm p-4 flex justify-between items-center z-10">
      <h1 className="text-xl font-bold text-gray-800 font-poppins">V-Match</h1>
      <div className="flex space-x-4 text-gray-600">
        <span>🔍</span>
        <span>👤</span>
        <span>🔔</span>
      </div>
    </header>
  );
};

export default Header;
