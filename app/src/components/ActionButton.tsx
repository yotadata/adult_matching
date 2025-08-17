import React from 'react';

interface ActionButtonProps {
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
}

const ActionButton: React.FC<ActionButtonProps> = ({ onClick, children, className }) => {
  return (
    <button
      onClick={onClick}
      className={`w-16 h-16 rounded-full flex items-center justify-center shadow-lg ${className}`}
    >
      {children}
    </button>
  );
};

export default ActionButton;
