import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface StopButtonProps {
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}

export function StopButton({ onClick, disabled = false, className }: StopButtonProps) {
  return (
    <Button
      data-testid="stop-button"
      type="button"
      variant="destructive"
      onClick={onClick}
      disabled={disabled}
      className={cn('gap-2', className)}
    >
      <StopIcon className="h-4 w-4" />
      Stop Research
    </Button>
  );
}

function StopIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
    >
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </svg>
  );
}
