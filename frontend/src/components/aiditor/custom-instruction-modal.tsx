/**
 * Modal for entering custom instructions for the CUSTOM highlight type.
 */

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';

interface CustomInstructionModalProps {
  open: boolean;
  selectedText: string;
  onConfirm: (instruction: string) => void;
  onCancel: () => void;
}

export function CustomInstructionModal({
  open,
  selectedText,
  onConfirm,
  onCancel,
}: CustomInstructionModalProps) {
  const [instruction, setInstruction] = useState('');

  const handleConfirm = () => {
    if (instruction.trim()) {
      onConfirm(instruction.trim());
      setInstruction('');
    }
  };

  const handleCancel = () => {
    setInstruction('');
    onCancel();
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Custom Edit Instruction</DialogTitle>
          <DialogDescription>
            Enter your custom instruction for how the AI should modify the
            selected text.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Selected Text Preview */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Selected Text</Label>
            <div className="p-3 bg-muted rounded-md text-sm max-h-24 overflow-y-auto">
              {selectedText || '(No text selected)'}
            </div>
          </div>

          {/* Instruction Input */}
          <div className="space-y-2">
            <Label htmlFor="instruction" className="text-sm font-medium">
              Instruction
            </Label>
            <Textarea
              id="instruction"
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder="e.g., Make it more formal, Add technical details, Simplify for beginners..."
              className="min-h-[100px] resize-none"
              maxLength={500}
              autoFocus
            />
            <p className="text-xs text-muted-foreground text-right">
              {instruction.length}/500 characters
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={!instruction.trim()}
          >
            Apply Highlight
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
