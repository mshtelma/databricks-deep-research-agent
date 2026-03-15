import { cn } from '@/lib/utils';

export interface QueryClassification {
  type: 'simple' | 'moderate' | 'complex' | 'clarification_needed' | 'follow_up';
  complexity: string;
  reasoning?: string;
  recommended_depth?: string;
}

interface ClassificationBadgeProps {
  classification: QueryClassification;
  showReasoning?: boolean;
  className?: string;
}

const TYPE_STYLES: Record<QueryClassification['type'], string> = {
  simple: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  moderate: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  complex: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
  clarification_needed: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
  follow_up: 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200',
};

const TYPE_LABELS: Record<QueryClassification['type'], string> = {
  simple: 'Simple Query',
  moderate: 'Moderate',
  complex: 'Complex Research',
  clarification_needed: 'Needs Clarification',
  follow_up: 'Follow-up',
};

export function ClassificationBadge({
  classification,
  showReasoning = false,
  className,
}: ClassificationBadgeProps) {
  const style = TYPE_STYLES[classification.type] || TYPE_STYLES.moderate;
  const label = TYPE_LABELS[classification.type] || classification.type;

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
            style
          )}
        >
          {label}
        </span>
        {classification.recommended_depth && (
          <span className="text-xs text-muted-foreground">
            Depth: {classification.recommended_depth}
          </span>
        )}
      </div>
      {showReasoning && classification.reasoning && (
        <p className="text-xs text-muted-foreground italic">
          {classification.reasoning}
        </p>
      )}
    </div>
  );
}
