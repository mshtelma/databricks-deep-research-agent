import type { AST } from '@/types/ast';
import { normalizeWorkflowAst } from '@/lib/workflowAst';

export interface DesignerSaveIdentity {
  isNew: boolean;
  agentName?: string | null;
  localName?: string | null;
  agentDescription?: string | null;
  localDescription?: string | null;
}

export interface DesignerSavePayload {
  definition: AST;
  name: string;
  description: string | null;
}

function nonBlank(value?: string | null): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function isDefaultName(value: string | null): boolean {
  return value === null || value === 'Untitled Agent';
}

export function buildDesignerSavePayload(
  rawAst: AST,
  identity: DesignerSaveIdentity,
): DesignerSavePayload {
  const displayName = nonBlank(identity.isNew ? identity.localName : identity.agentName);
  const alternateName = nonBlank(identity.isNew ? identity.agentName : identity.localName);
  const fallbackName = displayName ?? alternateName ?? 'Untitled Agent';
  const normalized = normalizeWorkflowAst(rawAst, fallbackName);

  const astName = nonBlank(normalized.name);
  const formDescription = nonBlank(
    identity.isNew ? identity.localDescription : identity.agentDescription,
  );
  const alternateDescription = nonBlank(
    identity.isNew ? identity.agentDescription : identity.localDescription,
  );
  const astDescription = nonBlank(normalized.description);

  const explicitNewName = identity.isNew && !isDefaultName(displayName) ? displayName : null;
  const definitionName = explicitNewName ?? astName ?? fallbackName;
  const definitionDescription = astDescription ?? formDescription ?? alternateDescription ?? '';

  const shouldUseDefinitionName = isDefaultName(displayName);
  const metadataName = shouldUseDefinitionName ? definitionName : (displayName ?? definitionName);
  const metadataDescription = formDescription ?? alternateDescription ?? astDescription;

  return {
    definition: {
      ...normalized,
      name: definitionName,
      description: definitionDescription,
    },
    name: metadataName,
    description: metadataDescription,
  };
}
