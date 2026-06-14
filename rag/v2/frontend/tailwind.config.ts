import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'selector',
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark theme design tokens
        surface: {
          DEFAULT: '#0f1117',
          card:    '#1a1d27',
          border:  '#2d3048',
        },
        accent: {
          DEFAULT: '#4f6ef7',
          hover:   '#3d5de6',
        },
      },
    },
  },
  plugins: [],
}
export default config
