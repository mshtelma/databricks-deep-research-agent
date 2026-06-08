import * as React from 'react';
import * as RadixTooltip from '@radix-ui/react-tooltip';
import { Zap, FileText } from 'lucide-react';

export interface ModeTabsProps {
  value: 'here' | 'other';
  onChange: (v: 'here' | 'other') => void;
  /** Tabs in this array will be rendered disabled with an optional tooltip. */
  disabledTabs?: Array<'here' | 'other'>;
  /** Tooltip text keyed by tab id — shown when that tab is disabled. */
  disabledTooltips?: Partial<Record<'here' | 'other', string>>;
}

const TABS: {
  id: 'here' | 'other';
  label: string;
  sub: string;
}[] = [
  { id: 'here', label: 'Deploy in this workspace', sub: 'One click · automatic' },
  { id: 'other', label: 'Export for another workspace', sub: 'Zip · manual steps' },
];

export function ModeTabs({ value, onChange, disabledTabs = [], disabledTooltips = {} }: ModeTabsProps) {
  return (
    <RadixTooltip.Provider delayDuration={200}>
      <div
        role="tablist"
        style={{
          display: 'flex',
          gap: 0,
          padding: 3,
          background: 'var(--db-oat-light)',
          borderRadius: 8,
          border: '1px solid var(--db-gray-lines)',
          alignSelf: 'stretch',
        }}
      >
        {TABS.map((tab) => {
          const active = value === tab.id;
          const disabled = disabledTabs.includes(tab.id);
          const tooltipText = disabledTooltips[tab.id];

          const btn = (
            <button
              key={tab.id}
              role="tab"
              aria-selected={active}
              aria-disabled={disabled || undefined}
              disabled={disabled}
              onClick={() => { if (!disabled) onChange(tab.id); }}
              style={{
                flex: 1,
                padding: '8px 12px',
                border: 0,
                cursor: disabled ? 'not-allowed' : 'pointer',
                borderRadius: 6,
                background: active ? '#fff' : 'transparent',
                boxShadow: active ? '0 1px 2px rgba(11,32,38,0.06)' : 'none',
                color: disabled
                  ? 'var(--db-gray-text)'
                  : active
                    ? 'var(--db-navy-800)'
                    : 'var(--db-gray-text)',
                opacity: disabled ? 0.5 : 1,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                textAlign: 'left',
                transition: 'all 120ms cubic-bezier(0.2,0.7,0.2,1)',
              }}
            >
              {tab.id === 'here' ? (
                <Zap
                  size={14}
                  color={active && !disabled ? 'var(--db-lava-600)' : 'var(--db-gray-text)'}
                />
              ) : (
                <FileText
                  size={14}
                  color={active && !disabled ? 'var(--db-lava-600)' : 'var(--db-gray-text)'}
                />
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <span style={{ fontSize: 12, fontWeight: 500, lineHeight: 1.2 }}>
                  {tab.label}
                </span>
                <span
                  style={{
                    fontSize: 10,
                    color: 'var(--db-gray-text)',
                    lineHeight: 1.2,
                  }}
                >
                  {tab.sub}
                </span>
              </div>
            </button>
          );

          if (disabled && tooltipText) {
            return (
              <RadixTooltip.Root key={tab.id}>
                <RadixTooltip.Trigger asChild>
                  <span style={{ flex: 1, display: 'flex' }}>{btn}</span>
                </RadixTooltip.Trigger>
                <RadixTooltip.Portal>
                  <RadixTooltip.Content
                    side="top"
                    style={{
                      background: 'var(--db-navy-900)',
                      color: '#fff',
                      fontSize: 11,
                      padding: '4px 8px',
                      borderRadius: 4,
                      maxWidth: 260,
                      lineHeight: 1.4,
                    }}
                  >
                    {tooltipText}
                    <RadixTooltip.Arrow style={{ fill: 'var(--db-navy-900)' }} />
                  </RadixTooltip.Content>
                </RadixTooltip.Portal>
              </RadixTooltip.Root>
            );
          }

          return <React.Fragment key={tab.id}>{btn}</React.Fragment>;
        })}
      </div>
    </RadixTooltip.Provider>
  );
}
