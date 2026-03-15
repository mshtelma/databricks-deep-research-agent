/**
 * Dialog for converting an incognito chat to a permanent chat.
 *
 * Provides a confirmation dialog explaining what "keeping" means
 * and handles the conversion mutation.
 */

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useConvertToRegular } from '@/hooks/useIncognitoChats'

interface KeepChatDialogProps {
  chatId: string
  onSuccess?: () => void
  className?: string
}

export function KeepChatDialog({
  chatId,
  onSuccess,
  className,
}: KeepChatDialogProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const convertMutation = useConvertToRegular()

  const handleKeep = async () => {
    try {
      await convertMutation.mutateAsync(chatId)
      setIsOpen(false)
      onSuccess?.()
    } catch (error) {
      console.error('Failed to convert chat:', error)
    }
  }

  return (
    <div className={cn('relative', className)}>
      {/* Trigger button */}
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(true)}
        className="gap-1.5"
      >
        <SaveIcon className="w-4 h-4" />
        Keep Chat
      </Button>

      {/* Dialog overlay and content */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setIsOpen(false)}
          />

          {/* Dialog */}
          <div
            className={cn(
              'relative z-10 w-full max-w-md mx-4',
              'bg-background rounded-lg shadow-lg border',
              'animate-in fade-in-0 zoom-in-95'
            )}
          >
            <div className="p-6">
              <h2 className="text-lg font-semibold mb-2">Keep this chat?</h2>
              <p className="text-muted-foreground text-sm mb-4">
                Converting this incognito chat to a regular chat will:
              </p>
              <ul className="text-sm space-y-2 mb-6 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <CheckIcon className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                  <span>Save the chat permanently to your account</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckIcon className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                  <span>Keep all messages and research results</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckIcon className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                  <span>Make it visible in your regular chat list</span>
                </li>
              </ul>

              <div className="flex gap-3 justify-end">
                <Button
                  variant="outline"
                  onClick={() => setIsOpen(false)}
                  disabled={convertMutation.isPending}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleKeep}
                  disabled={convertMutation.isPending}
                >
                  {convertMutation.isPending ? (
                    <>
                      <SpinnerIcon className="w-4 h-4 mr-2 animate-spin" />
                      Converting...
                    </>
                  ) : (
                    'Keep Chat'
                  )}
                </Button>
              </div>

              {convertMutation.isError && (
                <p className="mt-3 text-sm text-destructive">
                  Failed to convert chat. Please try again.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function SaveIcon({ className }: { className?: string }) {
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
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  )
}

function CheckIcon({ className }: { className?: string }) {
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
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function SpinnerIcon({ className }: { className?: string }) {
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
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  )
}

export default KeepChatDialog
