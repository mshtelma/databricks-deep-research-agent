import * as React from 'react';
import type { ToolDecl } from '@/types/ast';
import type { RegistryResponse } from '@/types/agentDesigner';
import { ToolDeclarationDialog } from './toolPicker/ToolDeclarationDialog';

export interface AddToolDialogProps {
  registry: RegistryResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeclared?: (tool: ToolDecl) => void;
}

export function AddToolDialog(props: AddToolDialogProps): React.ReactElement {
  return <ToolDeclarationDialog {...props} />;
}
