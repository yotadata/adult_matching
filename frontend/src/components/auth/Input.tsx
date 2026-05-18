'use client';

import { forwardRef, InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(({ label, id, ...props }, ref) => {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-[#8b949e] mb-1">
        {label}
      </label>
      <input
        id={id}
        ref={ref}
        {...props}
        className="w-full px-4 py-2.5 bg-[#161b22] rounded-lg border border-[#30363d] text-[#e6edf3] placeholder-[#484f58] focus:outline-none focus:ring-0 focus:border-violet-500 transition-all duration-300"
      />
    </div>
  );
});

Input.displayName = 'Input';

export default Input;
