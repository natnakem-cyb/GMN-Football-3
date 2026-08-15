/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        pitch: {
          dark: '#1e3a29',
          light: '#274e36',
          stripe: '#22442e',
          line: 'rgba(255, 255, 255, 0.85)',
        },
      },
    },
  },
  plugins: [],
};
