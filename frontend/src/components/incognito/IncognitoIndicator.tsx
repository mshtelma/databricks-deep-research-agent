/**
 * Incognito mode indicator component.
 *
 * Displays a subtle indicator that the current chat is in incognito mode,
 * with a tooltip explaining the behavior.
 */

import * as React from 'react'
import { cn } from '@/lib/utils'

interface IncognitoIndicatorProps {
  className?: string
  /** Show extended label vs icon only */
  showLabel?: boolean
}

export function IncognitoIndicator({
  className,
  showLabel = true,
}: IncognitoIndicatorProps) {
  const [showTooltip, setShowTooltip] = React.useState(false)

  return (
    <div
      className={cn('relative inline-flex items-center gap-1.5', className)}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div
        className={cn(
          'flex items-center gap-1.5 px-2 py-1 rounded-md',
          'bg-amber-100 dark:bg-amber-900/30',
          'text-amber-700 dark:text-amber-400',
          'text-xs font-medium'
        )}
      >
        <EyeOffIcon className="w-3.5 h-3.5" />
        {showLabel && <span>Incognito</span>}
      </div>

      {/* Tooltip */}
      {showTooltip && (
        <div
          className={cn(
            'absolute left-0 top-full mt-2 z-50',
            'w-64 p-3 rounded-lg shadow-lg',
            'bg-popover text-popover-foreground border',
            'text-xs animate-in fade-in-0 zoom-in-95'
          )}
        >
          <p className="font-medium mb-1">Incognito Chat</p>
          <p className="text-muted-foreground">
            This chat will be automatically deleted when your session expires
            (1 hour of inactivity). Click &quot;Keep Chat&quot; to save it permanently.
          </p>
        </div>
      )}
    </div>
  )
}

function EyeOffIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
      <path d="m1 1 22 22" />
      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
    </svg>
  )
}

export default IncognitoIndicator
