import { CodeBlock } from './CodeBlock';

export interface StepProps {
  n: number;
  title: string;
  body?: string;
  code?: string;
  codeLang?: string;
  codeLabel?: string;
  note?: string;
}

export function Step({ n, title, body, code, codeLang, codeLabel, note }: StepProps) {
  return (
    <li
      style={{
        display: 'flex',
        gap: 12,
        padding: '12px 0',
        borderBottom: '1px dashed var(--db-gray-lines)',
      }}
    >
      <span
        style={{
          flexShrink: 0,
          width: 22,
          height: 22,
          borderRadius: 999,
          background: 'var(--db-oat-medium)',
          color: 'var(--db-navy-800)',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 11,
          fontWeight: 600,
          fontFamily: 'var(--font-mono-db)',
        }}
      >
        {n}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: 'var(--db-navy-800)',
            lineHeight: 1.4,
          }}
        >
          {title}
        </div>
        {body && (
          <div
            style={{
              fontSize: 12,
              color: 'var(--db-gray-text)',
              marginTop: 3,
              lineHeight: 1.55,
            }}
          >
            {body}
          </div>
        )}
        {code && <CodeBlock code={code} lang={codeLang} label={codeLabel} />}
        {note && (
          <div
            style={{
              marginTop: 8,
              padding: '6px 10px',
              fontSize: 11,
              lineHeight: 1.5,
              background: 'rgba(255,219,150,0.25)',
              border: '1px solid var(--db-yellow-300, #FFDB96)',
              borderRadius: 4,
              color: 'var(--db-yellow-800, #7D5319)',
              display: 'flex',
              gap: 6,
              alignItems: 'flex-start',
            }}
          >
            <span style={{ fontSize: 11, flexShrink: 0 }}>ℹ</span>
            <span>{note}</span>
          </div>
        )}
      </div>
    </li>
  );
}
