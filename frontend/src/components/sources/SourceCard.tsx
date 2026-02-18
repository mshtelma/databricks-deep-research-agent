import * as React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type {
  DataSource,
  DataSourceType,
  DataSourceValidationStatus,
} from '@/types/dataSources'
import {
  getSourceTypeLabel,
  getCapabilityLabel,
  getValidationStatusColor,
  getValidationStatusLabel,
} from '@/types/dataSources'

interface SourceCardProps {
  source: DataSource
  onToggle?: (source: DataSource, enabled: boolean) => void
  onEdit?: (source: DataSource) => void
  onDelete?: (source: DataSource) => void
  onValidate?: (source: DataSource) => void
  onClick?: () => void
  isEnabled?: boolean
  isValidating?: boolean
  className?: string
}

/**
 * Card component for displaying a data source with its configuration and status.
 */
export function SourceCard({
  source,
  onToggle,
  onEdit,
  onDelete,
  onValidate,
  onClick,
  isEnabled = true,
  isValidating = false,
  className,
}: SourceCardProps) {
  const [showActions, setShowActions] = React.useState(false)

  return (
    <Card
      className={cn(
        'relative transition-shadow hover:shadow-md',
        !isEnabled && 'opacity-60',
        onClick && 'cursor-pointer',
        className
      )}
      onClick={onClick}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-3">
            <SourceTypeIcon type={source.type} className="h-8 w-8 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">{source.name}</CardTitle>
              <CardDescription className="text-xs mt-0.5">
                {getSourceTypeLabel(source.type)}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ValidationStatusBadge
              status={source.validation_status}
              onClick={() => onValidate?.(source)}
              isLoading={isValidating}
            />
            {onToggle && (
              <ToggleSwitch
                enabled={isEnabled}
                onChange={(enabled) => onToggle(source, enabled)}
              />
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {source.description && (
          <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
            {source.description}
          </p>
        )}

        {/* Capabilities */}
        {source.capabilities.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {source.capabilities.slice(0, 4).map((cap) => (
              <Badge key={cap} variant="secondary" className="text-xs">
                {getCapabilityLabel(cap)}
              </Badge>
            ))}
            {source.capabilities.length > 4 && (
              <Badge variant="outline" className="text-xs">
                +{source.capabilities.length - 4} more
              </Badge>
            )}
          </div>
        )}

        {/* Source-specific info */}
        <SourceDetails source={source} />

        {/* Action buttons - shown on hover */}
        {showActions && (onEdit || onDelete) && (
          <div className="absolute bottom-3 right-3 flex gap-2 animate-in fade-in-0 duration-150">
            {onEdit && (
              <Button size="sm" variant="outline" onClick={() => onEdit(source)}>
                <EditIcon className="h-3.5 w-3.5 mr-1" />
                Edit
              </Button>
            )}
            {onDelete && (
              <Button size="sm" variant="outline" onClick={() => onDelete(source)}>
                <TrashIcon className="h-3.5 w-3.5 mr-1" />
                Delete
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/**
 * Icon component for different source types.
 */
function SourceTypeIcon({ type, className }: { type: DataSourceType; className?: string }) {
  switch (type) {
    case 'vector_search':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v4" />
          <path d="M12 18v4" />
          <path d="M4.93 4.93l2.83 2.83" />
          <path d="M16.24 16.24l2.83 2.83" />
          <path d="M2 12h4" />
          <path d="M18 12h4" />
          <path d="M4.93 19.07l2.83-2.83" />
          <path d="M16.24 7.76l2.83-2.83" />
        </svg>
      )
    case 'genie':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
          <line x1="4" x2="4" y1="22" y2="15" />
        </svg>
      )
    case 'knowledge_assistant':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4" />
          <path d="M12 8h.01" />
        </svg>
      )
    case 'web_search':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="2" x2="22" y1="12" y2="12" />
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
      )
    case 'uploaded_file':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14,2 14,8 20,8" />
        </svg>
      )
    default:
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect width="18" height="18" x="3" y="3" rx="2" />
          <path d="M9 9h6" />
          <path d="M9 15h6" />
        </svg>
      )
  }
}

/**
 * Validation status badge with color coding.
 */
function ValidationStatusBadge({
  status,
  onClick,
  isLoading,
}: {
  status: DataSourceValidationStatus
  onClick?: () => void
  isLoading?: boolean
}) {
  const color = getValidationStatusColor(status)
  const label = getValidationStatusLabel(status)

  const colorClasses: Record<string, string> = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400',
    yellow: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-400',
    gray: 'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-400',
  }

  return (
    <button
      onClick={onClick}
      disabled={isLoading}
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-opacity',
        colorClasses[color],
        onClick && 'cursor-pointer hover:opacity-80',
        isLoading && 'opacity-50 cursor-wait'
      )}
      title={onClick ? 'Click to re-validate' : undefined}
    >
      {isLoading ? (
        <LoadingSpinner className="h-3 w-3 mr-1" />
      ) : (
        <StatusDot color={color} className="mr-1" />
      )}
      {label}
    </button>
  )
}

/**
 * Small status indicator dot.
 */
function StatusDot({ color, className }: { color: string; className?: string }) {
  const bgColors: Record<string, string> = {
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    gray: 'bg-gray-500',
  }

  return <span className={cn('h-2 w-2 rounded-full', bgColors[color], className)} />
}

/**
 * Toggle switch component.
 */
function ToggleSwitch({
  enabled,
  onChange,
}: {
  enabled: boolean
  onChange: (enabled: boolean) => void
}) {
  return (
    <button
      role="switch"
      aria-checked={enabled}
      onClick={() => onChange(!enabled)}
      className={cn(
        'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent',
        'transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        enabled ? 'bg-primary' : 'bg-muted'
      )}
    >
      <span
        className={cn(
          'pointer-events-none block h-4 w-4 rounded-full bg-background shadow-lg',
          'transition-transform',
          enabled ? 'translate-x-4' : 'translate-x-0'
        )}
      />
    </button>
  )
}

/**
 * Source-specific details display.
 */
function SourceDetails({ source }: { source: DataSource }) {
  const { config, type } = source

  // Type-safe access to config properties
  const indexName = config.index_name as string | undefined
  const spaceId = config.space_id as string | undefined
  const endpointName = config.endpoint_name as string | undefined

  return (
    <div className="text-xs text-muted-foreground space-y-1">
      {type === 'vector_search' && indexName && (
        <div className="flex items-center gap-1">
          <span className="font-medium">Index:</span>
          <code className="bg-muted px-1 rounded text-[10px]">{indexName}</code>
        </div>
      )}
      {type === 'genie' && spaceId && (
        <div className="flex items-center gap-1">
          <span className="font-medium">Space:</span>
          <code className="bg-muted px-1 rounded text-[10px]">{spaceId}</code>
        </div>
      )}
      {type === 'knowledge_assistant' && endpointName && (
        <div className="flex items-center gap-1">
          <span className="font-medium">Endpoint:</span>
          <code className="bg-muted px-1 rounded text-[10px]">{endpointName}</code>
        </div>
      )}
    </div>
  )
}

/**
 * Loading spinner component.
 */
function LoadingSpinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn('animate-spin', className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="m4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

/**
 * Edit icon component.
 */
function EditIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
      <path d="m15 5 4 4" />
    </svg>
  )
}

/**
 * Trash icon component.
 */
function TrashIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  )
}
