/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  prefix: '',
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      colors: {
        // Databricks brand colors
        databricks: {
          orange: '#FF6F00',
          blue: '#0072C3',
          'dark-blue': '#003D82',
          'light-orange': '#FFB84D',
          'light-blue': '#B3D9FF',
        },

        // Agent-specific colors
        agent: {
          coordinator: {
            DEFAULT: '#0072C3',
            light: '#B3D9FF',
            dark: '#003D82',
          },
          planner: {
            DEFAULT: '#7C3AED',
            light: '#DDD6FE',
            dark: '#5B21B6',
          },
          researcher: {
            DEFAULT: '#FF6F00',
            light: '#FFD6B3',
            dark: '#CC5600',
          },
          checker: {
            DEFAULT: '#DC2626',
            light: '#FECACA',
            dark: '#991B1B',
          },
          reporter: {
            DEFAULT: '#059669',
            light: '#A7F3D0',
            dark: '#047857',
          },
        },

        // UI System colors
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'thinking-dots': {
          '0%, 80%, 100%': {
            transform: 'scale(0.8)',
            opacity: '0.5',
          },
          '40%': {
            transform: 'scale(1)',
            opacity: '1',
          },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'thinking-dots': 'thinking-dots 1.4s infinite ease-in-out both',
      },
    },
  },
  plugins: [require('tailwindcss-animate'), require('@tailwindcss/typography')],
}