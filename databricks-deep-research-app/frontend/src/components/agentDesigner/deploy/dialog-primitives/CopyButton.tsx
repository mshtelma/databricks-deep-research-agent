import { useState } from 'react';

export interface CopyButtonProps {
  text: string;
  label?: string;
}

export function CopyButton({ text, label = 'Copy' }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  function handleClick(e: React.MouseEvent<HTMLButtonElement>) {
    e.stopPropagation();
    try {
      navigator.clipboard.writeText(text);
    } catch {
      // clipboard may be unavailable in some contexts
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1100);
  }

  return (
    <button
      onClick={handleClick}
      className="inline-flex items-center cursor-pointer border-0 rounded"
      style={{
        background: 'var(--db-navy-800)',
        color: copied ? 'var(--db-green-300, #9ED6C4)' : '#fff',
        padding: '3px 8px',
        fontSize: 10,
        fontFamily: 'var(--font-mono-db)',
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        transition: 'color 120ms ease-out',
      }}
    >
      {copied ? '✓ Copied' : label}
    </button>
  );
}
