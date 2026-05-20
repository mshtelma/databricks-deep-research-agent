import { type ReactNode } from 'react';
import * as RadixDialog from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { type LucideIcon } from 'lucide-react';

export interface DialogShellProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  icon: LucideIcon;
  iconBg: string;
  iconColor: string;
  title: string;
  subtitle: string;
  width?: number;
  children: ReactNode;
  footer?: ReactNode;
}

export function DialogShell({
  open,
  onOpenChange,
  icon: Icon,
  iconBg,
  iconColor,
  title,
  subtitle,
  width = 720,
  children,
  footer,
}: DialogShellProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        {/* Scrim — mirrors .scrim: fixed inset-0, navy/30 bg, blur-2, 200ms fade */}
        <RadixDialog.Overlay
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(11,32,38,0.30)',
            backdropFilter: 'blur(2px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 50,
            animation: 'blockIn 200ms cubic-bezier(0.2,0.7,0.2,1)',
          }}
        />
        <RadixDialog.Content
          style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: '#fff',
            borderRadius: 12,
            width,
            maxWidth: 'calc(100vw - 48px)',
            maxHeight: 'calc(100vh - 48px)',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 24px 64px rgba(11,32,38,0.18)',
            overflow: 'hidden',
            zIndex: 51,
            animation: 'blockIn 200ms cubic-bezier(0.2,0.7,0.2,1)',
            outline: 'none',
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '18px 22px 14px',
              borderBottom: '1px solid var(--db-gray-lines)',
              flexShrink: 0,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
              <span
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 8,
                  background: iconBg,
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <Icon size={18} color={iconColor} />
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <RadixDialog.Title
                  style={{
                    font: '500 17px/1.3 var(--font-sans-db)',
                    margin: 0,
                    color: 'var(--db-navy-800)',
                  }}
                >
                  {title}
                </RadixDialog.Title>
                <RadixDialog.Description
                  style={{
                    fontSize: 12,
                    color: 'var(--db-gray-text)',
                    margin: '4px 0 0',
                    lineHeight: 1.55,
                  }}
                >
                  {subtitle}
                </RadixDialog.Description>
              </div>
              <RadixDialog.Close
                style={{
                  background: 'transparent',
                  color: 'var(--db-gray-text)',
                  border: 0,
                  padding: '4px 6px',
                  borderRadius: 4,
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  flexShrink: 0,
                }}
              >
                <X size={14} />
              </RadixDialog.Close>
            </div>
          </div>

          {/* Scrollable body */}
          <div style={{ flex: 1, overflow: 'auto', padding: '16px 22px' }}>
            {children}
          </div>

          {/* Optional sticky footer */}
          {footer && (
            <div
              style={{
                padding: '12px 22px',
                borderTop: '1px solid var(--db-gray-lines)',
                background: 'var(--db-oat-light)',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                flexShrink: 0,
              }}
            >
              {footer}
            </div>
          )}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}
