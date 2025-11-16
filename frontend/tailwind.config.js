/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0a0a0a',
          surface: '#1a1a1a',
          card: '#242424',
          border: '#333333',
          text: '#ffffff',
          'text-muted': '#a0a0a0',
        },
        light: {
          bg: '#ffffff',
          surface: '#f5f5f5',
          card: '#ffffff',
          border: '#e5e5e5',
          text: '#1a1a1a',
          'text-muted': '#666666',
        }
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}

