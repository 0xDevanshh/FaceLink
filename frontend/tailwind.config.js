/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark OSINT theme palette
        surface: {
          DEFAULT: '#0f1117',
          1: '#161b22',
          2: '#1c2333',
          3: '#21262d',
        },
        accent: {
          DEFAULT: '#58a6ff',
          dim: '#388bfd',
        },
        success: '#3fb950',
        warn: '#d29922',
        danger: '#f85149',
        muted: '#8b949e',
        border: '#30363d',
      },
    },
  },
  plugins: [],
}
