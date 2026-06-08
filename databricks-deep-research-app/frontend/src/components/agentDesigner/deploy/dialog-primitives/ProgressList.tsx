import { Check, X } from 'lucide-react';

export interface ProgressStep {
  id: string;
  label: string;
  detail?: string;
}

export interface ProgressListProps {
  steps: ProgressStep[];
  currentIdx: number;
  error?: boolean;
}

export function ProgressList({ steps, currentIdx, error = false }: ProgressListProps) {
  return (
    <ul
      style={{
        listStyle: 'none',
        padding: 0,
        margin: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      {steps.map((s, i) => {
        const done = i < currentIdx;
        const live = i === currentIdx && !error;
        const failed = i === currentIdx && error;
        const idle = i > currentIdx;

        return (
          <li
            key={s.id}
            style={{
              display: 'flex',
              gap: 10,
              padding: '8px 10px',
              borderRadius: 6,
              background: live ? 'var(--db-oat-light)' : 'transparent',
              alignItems: 'flex-start',
            }}
          >
            <span
              className={live ? 'db-anim-pulseRing' : undefined}
              style={{
                width: 18,
                height: 18,
                marginTop: 1,
                flexShrink: 0,
                borderRadius: 999,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: done
                  ? 'var(--db-green-700)'
                  : failed
                  ? 'var(--db-lava-600)'
                  : live
                  ? '#fff'
                  : 'var(--db-oat-medium)',
                border: live ? '2px solid var(--db-lava-600)' : 'none',
              }}
            >
              {done && <Check size={11} color="#fff" strokeWidth={3} />}
              {failed && <X size={10} color="#fff" strokeWidth={3} />}
            </span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: 12,
                  fontWeight: live || failed ? 500 : 400,
                  color: idle ? 'var(--db-gray-text)' : 'var(--db-navy-800)',
                }}
              >
                {s.label}
              </div>
              {s.detail && (done || live || failed) && (
                <div
                  style={{
                    fontSize: 11,
                    color: 'var(--db-gray-text)',
                    marginTop: 2,
                    fontFamily: 'var(--font-mono-db)',
                  }}
                >
                  {s.detail}
                </div>
              )}
            </div>
            {live && (
              <span
                style={{
                  fontSize: 10,
                  color: 'var(--db-lava-600)',
                  fontFamily: 'var(--font-mono-db)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  flexShrink: 0,
                }}
              >
                running
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
