import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { templatesApi } from '../api/templates';
import type {
  Template,
  TemplateListParams,
  CreateTemplateRequest,
  UpdateTemplateRequest,
  RenderTemplateRequest,
  TemplateType,
} from '../types/templates';

const TEMPLATES_KEY = ['templates'];

/**
 * Hook to fetch all templates accessible to the user.
 */
export function useTemplates(params?: TemplateListParams) {
  return useQuery({
    queryKey: [...TEMPLATES_KEY, params],
    queryFn: () => templatesApi.list(params),
    gcTime: Infinity,
  });
}

/**
 * Hook to fetch a single template by ID.
 */
export function useTemplate(templateId: string | undefined) {
  return useQuery({
    queryKey: [...TEMPLATES_KEY, templateId],
    queryFn: () => (templateId ? templatesApi.get(templateId) : null),
    enabled: !!templateId,
    gcTime: Infinity,
  });
}

/**
 * Hook to create a new template.
 */
export function useCreateTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateTemplateRequest) => templatesApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: TEMPLATES_KEY });
    },
  });
}

/**
 * Hook to update an existing template.
 */
export function useUpdateTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ templateId, data }: { templateId: string; data: UpdateTemplateRequest }) =>
      templatesApi.update(templateId, data),
    onSuccess: (_, { templateId }) => {
      queryClient.invalidateQueries({ queryKey: TEMPLATES_KEY });
      queryClient.invalidateQueries({ queryKey: [...TEMPLATES_KEY, templateId] });
    },
  });
}

/**
 * Hook to delete a template.
 */
export function useDeleteTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (templateId: string) => templatesApi.delete(templateId),
    onSuccess: (_, templateId) => {
      queryClient.invalidateQueries({ queryKey: TEMPLATES_KEY });
      queryClient.removeQueries({ queryKey: [...TEMPLATES_KEY, templateId] });
    },
  });
}

/**
 * Hook to render a template with provided variables.
 */
export function useRenderTemplate() {
  return useMutation({
    mutationFn: (data: RenderTemplateRequest) => templatesApi.render(data),
  });
}

/**
 * Hook to get the default template for a specific type.
 */
export function useDefaultTemplate(type: TemplateType | undefined) {
  return useQuery({
    queryKey: [...TEMPLATES_KEY, 'default', type],
    queryFn: () => (type ? templatesApi.getDefault(type) : null),
    enabled: !!type,
    gcTime: Infinity,
  });
}

/**
 * Hook to set a template as the default for its type.
 */
export function useSetDefaultTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (templateId: string) => templatesApi.setDefault(templateId),
    onSuccess: (updatedTemplate) => {
      queryClient.invalidateQueries({ queryKey: TEMPLATES_KEY });
      // Also invalidate the default template query for this type
      if (updatedTemplate) {
        queryClient.invalidateQueries({
          queryKey: [...TEMPLATES_KEY, 'default', updatedTemplate.type],
        });
      }
    },
  });
}

/**
 * Hook to clone an existing template.
 */
export function useCloneTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ templateId, newName }: { templateId: string; newName?: string }) =>
      templatesApi.clone(templateId, newName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: TEMPLATES_KEY });
    },
  });
}

/**
 * Convenience hook that returns templates grouped by type.
 * Useful for the TemplateLibrary component.
 */
export function useGroupedTemplates(params?: TemplateListParams) {
  const { data, isLoading, error } = useTemplates(params);

  const grouped = {
    system: [] as Template[],
    step: [] as Template[],
    synthesis: [] as Template[],
    query: [] as Template[],
  };

  if (data?.templates) {
    for (const template of data.templates) {
      if (template.type in grouped) {
        grouped[template.type].push(template);
      }
    }
  }

  return {
    grouped,
    templates: data?.templates ?? [],
    total: data?.total ?? 0,
    userTemplates: data?.userTemplates ?? 0,
    workspaceTemplates: data?.workspaceTemplates ?? 0,
    isLoading,
    error,
  };
}

// Export query key for external use
export { TEMPLATES_KEY };
