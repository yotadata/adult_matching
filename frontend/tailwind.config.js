/** @type {import('tailwindcss').Config} */
const config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        darkAccent: '#18186A',
      },
    },
  },
  plugins: [],
};
module.exports = config;