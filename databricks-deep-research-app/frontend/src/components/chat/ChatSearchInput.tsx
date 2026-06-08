import * as React from 'react';
import { cn } from '@/lib/utils';

interface ChatSearchInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function ChatSearchInput({
  value,
  onChange,
  placeholder = 'Search chats',
  className,
}: ChatSearchInputProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Focus on Cmd/Ctrl + K
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div
      className={cn(
        'flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-db-oat-light px-2.5 py-1.5 transition-colors focus-within:border-db-navy-400 focus-within:shadow-db-focus',
        className,
      )}
    >
      <SearchIcon className="h-3 w-3 shrink-0 text-db-navy-400" />
      <input
        ref={inputRef}
        type="text"
        data-testid="chat-search-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="flex-1 border-0 bg-transparent text-[12px] font-normal text-db-navy-800 outline-none placeholder:text-db-gray-text"
      />
      {value && (
        <button
          type="button"
          data-testid="chat-search-clear"
          onClick={() => onChange('')}
          className="text-db-gray-text hover:text-db-navy-800"
          aria-label="Clear search"
        >
          <ClearIcon className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}

function SearchIcon({ className }: { className?: string }) {
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
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function ClearIcon({ className }: { className?: string }) {
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
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}
