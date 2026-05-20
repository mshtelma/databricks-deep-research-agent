import { CopyButton } from './CopyButton';

export interface CodeBlockProps {
  code: string;
  lang?: string;
  label?: string;
  multiline?: boolean;
}

export function CodeBlock({ code, lang, label, multiline = true }: CodeBlockProps) {
  return (
    <div
      style={{
        background: 'var(--db-navy-900)',
        borderRadius: 6,
        marginTop: 6,
        fontFamily: 'var(--font-mono-db)',
        fontSize: 12,
        color: '#E6ECEF',
        overflow: 'hidden',
        border: '1px solid var(--db-navy-800)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 10px 6px 12px',
          borderBottom: '1px solid var(--db-navy-800)',
          background: 'rgba(255,255,255,0.03)',
        }}
      >
        <span
          style={{
            fontSize: 10,
            color: 'var(--db-navy-300)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          {label ?? lang ?? 'shell'}
        </span>
        <CopyButton text={code} />
      </div>
      <pre
        style={{
          margin: 0,
          padding: '10px 12px',
          whiteSpace: multiline ? 'pre' : 'pre-wrap',
          overflowX: 'auto',
          lineHeight: 1.55,
        }}
      >
        {code}
      </pre>
    </div>
  );
}
