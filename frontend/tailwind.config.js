/** @type {import('tailwindcss').Config} */
const config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          purple: '#A78BFA', // Nope/アクセント（紫）
          amber: '#FBBF24',  // Like/アクセント（黄）
        },
        gradient: {
          a: '#C4C8E3',
          b: '#D7D1E3',
          c: '#F7D7E0',
          d: '#F9C9D6',
        },
      },
      boxShadow: {
        glass: '0 20px 60px rgba(0,0,0,0.25)',
        btnPurple: '0 10px 20px rgba(167,139,250,0.35)',
        btnAmber: '0 10px 20px rgba(251,191,36,0.35)',
      },
      borderRadius: {
        xl: '1rem',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [require('@tailwindcss/aspect-ratio')],
};
module.exports = config;
