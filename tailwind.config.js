/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#070A12',
          panel: '#0B1220',
          border: '#1C2A44',
          text: '#E6F1FF',
          muted: '#A7B7D6',
          neon: '#22D3EE',
          neon2: '#3B82F6',
          danger: '#FB7185',
          warn: '#FBBF24',
          ok: '#34D399',
        },
      },
      boxShadow: {
        neon: '0 0 0 1px rgba(34,211,238,0.35), 0 0 30px rgba(34,211,238,0.15)',
        neonStrong:
          '0 0 0 1px rgba(34,211,238,0.6), 0 0 40px rgba(34,211,238,0.25)',
      },
      backgroundImage: {
        grid:
          'radial-gradient(circle at 1px 1px, rgba(34,211,238,0.12) 1px, transparent 0)',
        glow:
          'radial-gradient(600px circle at var(--x, 20%) var(--y, 20%), rgba(34,211,238,0.22), transparent 45%)',
      },
    },
  },
  plugins: [],
}

