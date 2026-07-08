import * as React from 'react';
import type { ToolDecl } from '@/types/ast';
import type { RegistryResponse } from '@/types/agentDesigner';
import {
  ToolDeclarationDialog,
  type AddToolIntent,
} from './toolPicker/ToolDeclarationDialog';

export type { AddToolIntent };

export interface AddToolDialogProps {
  registry: RegistryResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeclared?: (tool: ToolDecl) => void;
  /** Launch context — labels the primary action (Add / Add and enable / Add and call). */
  intent?: AddToolIntent;
  /** Pre-seed the picker's search box (e.g. converting a direct ref's FQN). */
  initialQuery?: string;
}

export function AddToolDialog(props: AddToolDialogProps): React.ReactElement {
  return <ToolDeclarationDialog {...props} />;
}
