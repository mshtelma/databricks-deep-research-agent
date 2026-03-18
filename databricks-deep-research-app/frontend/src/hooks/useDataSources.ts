import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { dataSourcesApi } from '../api/dataSources'
import type {
  CreateVectorSearchRequest,
  CreateGenieRequest,
  CreateKnowledgeAssistantRequest,
  UpdateDataSourceRequest,
  ValidateConnectionRequest,
} from '../types/dataSources'

const DATA_SOURCES_KEY = ['data-sources']

/**
 * Hook to fetch all data sources accessible to the user.
 */
export function useDataSources(params?: { visibility?: string; type?: string }) {
  return useQuery({
    queryKey: [...DATA_SOURCES_KEY, params],
    queryFn: () => dataSourcesApi.list(params),
    gcTime: Infinity,
  })
}

/**
 * Hook to fetch a single data source by ID.
 */
export function useDataSource(sourceId: string | undefined) {
  return useQuery({
    queryKey: [...DATA_SOURCES_KEY, sourceId],
    queryFn: () => (sourceId ? dataSourcesApi.get(sourceId) : null),
    enabled: !!sourceId,
    gcTime: Infinity,
  })
}

/**
 * Hook to create a Vector Search data source.
 */
export function useCreateVectorSearchSource() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateVectorSearchRequest) => dataSourcesApi.createVectorSearch(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DATA_SOURCES_KEY })
    },
  })
}

/**
 * Hook to create a Genie data source.
 */
export function useCreateGenieSource() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateGenieRequest) => dataSourcesApi.createGenie(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DATA_SOURCES_KEY })
    },
  })
}

/**
 * Hook to create a Knowledge Assistant data source.
 */
export function useCreateKnowledgeAssistantSource() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateKnowledgeAssistantRequest) =>
      dataSourcesApi.createKnowledgeAssistant(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DATA_SOURCES_KEY })
    },
  })
}

/**
 * Hook to update an existing data source.
 */
export function useUpdateDataSource() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ sourceId, data }: { sourceId: string; data: UpdateDataSourceRequest }) =>
      dataSourcesApi.update(sourceId, data),
    onSuccess: (_, { sourceId }) => {
      queryClient.invalidateQueries({ queryKey: DATA_SOURCES_KEY })
      queryClient.invalidateQueries({ queryKey: [...DATA_SOURCES_KEY, sourceId] })
    },
  })
}

/**
 * Hook to delete a data source.
 */
export function useDeleteDataSource() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (sourceId: string) => dataSourcesApi.delete(sourceId),
    onSuccess: (_, sourceId) => {
      queryClient.invalidateQueries({ queryKey: DATA_SOURCES_KEY })
      queryClient.removeQueries({ queryKey: [...DATA_SOURCES_KEY, sourceId] })
    },
  })
}

/**
 * Hook to validate a data source (check OBO access).
 */
export function useValidateDataSource() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (sourceId: string) => dataSourcesApi.validate(sourceId),
    onSuccess: (_, sourceId) => {
      // Invalidate the specific source to refresh validation status
      queryClient.invalidateQueries({ queryKey: [...DATA_SOURCES_KEY, sourceId] })
      // Also refresh the list to show updated status
      queryClient.invalidateQueries({ queryKey: DATA_SOURCES_KEY })
    },
  })
}

/**
 * Hook to validate connection parameters before creating a data source.
 */
export function useValidateConnection() {
  return useMutation({
    mutationFn: (data: ValidateConnectionRequest) => dataSourcesApi.validateConnection(data),
  })
}

/**
 * Convenience hook that returns data sources grouped by category.
 * Useful for the SourceBrowser component.
 */
export function useGroupedDataSources() {
  const { data, isLoading, error } = useDataSources()

  const grouped = {
    webSources: [] as import('../types/dataSources').DataSource[],
    vectorSearch: [] as import('../types/dataSources').DataSource[],
    genie: [] as import('../types/dataSources').DataSource[],
    knowledgeAssistants: [] as import('../types/dataSources').DataSource[],
    uploadedFiles: [] as import('../types/dataSources').DataSource[],
    custom: [] as import('../types/dataSources').DataSource[],
  }

  if (data?.sources) {
    for (const source of data.sources) {
      switch (source.type) {
        case 'web_search':
          grouped.webSources.push(source)
          break
        case 'vector_search':
          grouped.vectorSearch.push(source)
          break
        case 'genie':
          grouped.genie.push(source)
          break
        case 'knowledge_assistant':
          grouped.knowledgeAssistants.push(source)
          break
        case 'uploaded_file':
          grouped.uploadedFiles.push(source)
          break
        case 'custom':
          grouped.custom.push(source)
          break
      }
    }
  }

  return {
    grouped,
    sources: data?.sources ?? [],
    total: data?.total ?? 0,
    userSources: data?.user_sources ?? 0,
    workspaceSources: data?.workspace_sources ?? 0,
    isLoading,
    error,
  }
}

// Export query key for external use
export { DATA_SOURCES_KEY }
