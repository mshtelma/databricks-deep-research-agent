import { type CSSProperties, type ReactNode } from 'react';

export interface SectionTitleProps {
  children: ReactNode;
  style?: CSSProperties;
}

export function SectionTitle({ children, style }: SectionTitleProps) {
  return (
    <div
      style={{
        fontSize: 12,
        fontWeight: 500,
        color: 'var(--db-navy-800)',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        marginBottom: 6,
        ...style,
      }}
    >
      {children}
    </div>
  );
}
