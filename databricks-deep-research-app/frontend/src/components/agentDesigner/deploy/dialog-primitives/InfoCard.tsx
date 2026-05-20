import { type ReactNode } from 'react';

export type InfoCardColor = 'blue' | 'green' | 'yellow' | 'lava';

export interface InfoCardProps {
  color: InfoCardColor;
  children: ReactNode;
}

const COLOR_MAP: Record<
  InfoCardColor,
  { bg: string; border: string; text: string }
> = {
  blue: {
    bg: 'var(--db-blue-100)',
    border: 'var(--db-blue-300)',
    text: 'var(--db-blue-700)',
  },
  green: {
    bg: 'var(--db-green-300)',
    border: 'var(--db-green-700)',
    text: 'var(--db-navy-800)',
  },
  yellow: {
    bg: 'var(--db-yellow-300)',
    border: 'var(--db-yellow-700)',
    text: 'var(--db-yellow-800)',
  },
  lava: {
    bg: 'var(--db-lava-100)',
    border: 'var(--db-lava-300)',
    text: 'var(--db-lava-700)',
  },
};

export function InfoCard({ color, children }: InfoCardProps) {
  const c = COLOR_MAP[color];
  return (
    <div
      style={{
        marginTop: 14,
        padding: '10px 12px',
        borderRadius: 6,
        background: c.bg,
        border: `1px solid ${c.border}`,
        fontSize: 12,
        color: c.text,
        lineHeight: 1.6,
      }}
    >
      {children}
    </div>
  );
}
