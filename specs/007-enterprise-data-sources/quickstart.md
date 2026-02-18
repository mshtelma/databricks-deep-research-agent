# Quickstart: Enterprise Data Source Discovery

**Feature**: 007-enterprise-data-sources (US9a/US9b)
**Date**: 2026-02-04

---

## Overview

This guide explains how to implement and use the data source discovery feature, which automatically discovers Vector Search indexes, Genie spaces, and Knowledge Assistants available to the authenticated user.

---

## 1. Backend Implementation

### 1.1 Install Dependencies

Ensure you have the required packages:

```bash
uv add databricks-sdk databricks-ai-bridge
```

### 1.2 Create Discovery Service

Create `src/deep_research/services/discovery_service.py`:

```python
"""Data source discovery service using Databricks SDK."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from pydantic import BaseModel

from deep_research.schemas.discovery import (
    DiscoveredSource,
    DiscoveryResponse,
    VectorSearchMetadata,
    GenieSpaceMetadata,
    ServingEndpointMetadata,
    DataSourceType,
    DiscoveryStatus,
)


class DiscoveryService:
    """Service for discovering available data sources."""

    def __init__(self, cache_ttl: timedelta = timedelta(minutes=5)):
        self._cache: dict[str, tuple[list[DiscoveredSource], datetime]] = {}
        self._cache_ttl = cache_ttl
        self._lock = asyncio.Lock()

    async def discover_all(
        self,
        user_id: str,
        force_refresh: bool = False,
    ) -> DiscoveryResponse:
        """Discover all available data sources for the user."""
        cache_key = f"discovery:{user_id}"

        # Check cache
        if not force_refresh:
            async with self._lock:
                if cache_key in self._cache:
                    sources, cached_at = self._cache[cache_key]
                    if datetime.utcnow() - cached_at < self._cache_ttl:
                        return self._build_response(sources, cached=True, cached_at=cached_at)

        # Discover in parallel
        w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

        vs_task = asyncio.to_thread(self._discover_vector_search, w)
        genie_task = asyncio.to_thread(self._discover_genie, w)
        serving_task = asyncio.to_thread(self._discover_serving, w)

        results = await asyncio.gather(vs_task, genie_task, serving_task, return_exceptions=True)

        # Collect results
        sources = []
        errors = []
        for result, source_type in zip(results, [DataSourceType.VECTOR_SEARCH, DataSourceType.GENIE, DataSourceType.KNOWLEDGE_ASSISTANT]):
            if isinstance(result, Exception):
                errors.append({
                    "source_type": source_type,
                    "error_code": type(result).__name__,
                    "error_message": str(result),
                    "retryable": True,
                })
            else:
                sources.extend(result)

        # Update cache
        now = datetime.utcnow()
        async with self._lock:
            self._cache[cache_key] = (sources, now)

        return self._build_response(sources, cached=False, cached_at=now, errors=errors if errors else None)

    def _discover_vector_search(self, w: WorkspaceClient) -> list[DiscoveredSource]:
        """Discover Vector Search indexes."""
        sources = []

        for endpoint in w.vector_search_endpoints.list_endpoints():
            for mini_index in w.vector_search_indexes.list_indexes(endpoint.name):
                try:
                    index = w.vector_search_indexes.get_index(mini_index.name)

                    # Extract metadata
                    metadata = VectorSearchMetadata(
                        index_name=index.name,
                        endpoint_name=endpoint.name,
                        primary_key=index.primary_key or "id",
                        index_type=index.index_type.value if index.index_type else "UNKNOWN",
                        embedding_columns=self._extract_embedding_columns(index),
                        filter_columns=self._extract_filter_columns(index),
                        supported_query_types=self._get_supported_query_types(index),
                        supports_reranking=True,  # All VS indexes support reranking
                        is_ready=index.status.ready if index.status else False,
                    )

                    sources.append(DiscoveredSource(
                        source_id=f"vs:{index.name}",
                        source_type=DataSourceType.VECTOR_SEARCH,
                        name=index.name.split(".")[-1],  # Last part of qualified name
                        endpoint_name=endpoint.name,
                        description=None,  # VS indexes don't have descriptions
                        status=DiscoveryStatus.READY if metadata.is_ready else DiscoveryStatus.SYNCING,
                        capabilities=metadata.supported_query_types + (["reranking"] if metadata.supports_reranking else []),
                        metadata=metadata.model_dump(),
                        discovered_at=datetime.utcnow(),
                        cached_until=datetime.utcnow() + self._cache_ttl,
                    ))
                except Exception as e:
                    # Log and skip individual index errors
                    continue

        return sources

    def _discover_genie(self, w: WorkspaceClient) -> list[DiscoveredSource]:
        """Discover Genie spaces."""
        sources = []

        response = w.genie.list_spaces()
        for space_summary in response.spaces or []:
            try:
                space = w.genie.get_space(space_summary.id)

                metadata = GenieSpaceMetadata(
                    space_id=space_summary.id,
                    title=space.title or space_summary.id,
                    description=space.description,
                    warehouse_id=space.warehouse_id,
                    owner=space.creator,
                )

                sources.append(DiscoveredSource(
                    source_id=f"genie:{space_summary.id}",
                    source_type=DataSourceType.GENIE,
                    name=metadata.title,
                    endpoint_name=space_summary.id,
                    description=metadata.description,
                    status=DiscoveryStatus.READY,
                    capabilities=["sql", "conversation"],
                    metadata=metadata.model_dump(),
                    discovered_at=datetime.utcnow(),
                    cached_until=datetime.utcnow() + self._cache_ttl,
                ))
            except Exception:
                continue

        return sources

    def _discover_serving(self, w: WorkspaceClient) -> list[DiscoveredSource]:
        """Discover serving endpoints (Knowledge Assistants)."""
        sources = []

        for endpoint in w.serving_endpoints.list():
            # Filter for likely Knowledge Assistants
            if not self._is_knowledge_assistant(endpoint):
                continue

            metadata = ServingEndpointMetadata(
                endpoint_name=endpoint.name,
                endpoint_type=str(endpoint.endpoint_type) if endpoint.endpoint_type else "UNKNOWN",
                state=endpoint.state.value if endpoint.state else "UNKNOWN",
                tags=dict(endpoint.tags) if endpoint.tags else {},
                is_knowledge_assistant=True,
                creator=endpoint.creator,
            )

            sources.append(DiscoveredSource(
                source_id=f"assistant:{endpoint.name}",
                source_type=DataSourceType.KNOWLEDGE_ASSISTANT,
                name=endpoint.name,
                endpoint_name=endpoint.name,
                description=None,
                status=DiscoveryStatus.READY if endpoint.state and endpoint.state.value == "READY" else DiscoveryStatus.UNAVAILABLE,
                capabilities=["chat", "context"],
                metadata=metadata.model_dump(),
                discovered_at=datetime.utcnow(),
                cached_until=datetime.utcnow() + self._cache_ttl,
            ))

        return sources

    def _is_knowledge_assistant(self, endpoint) -> bool:
        """Heuristic to identify Knowledge Assistants."""
        name_lower = endpoint.name.lower()
        if any(kw in name_lower for kw in ["assistant", "expert", "helper", "advisor"]):
            return True

        if endpoint.tags:
            tags = {k.lower(): v.lower() for k, v in endpoint.tags.items()}
            if "assistant" in tags or "knowledge" in tags:
                return True

        return False

    def _extract_embedding_columns(self, index) -> list[str]:
        """Extract embedding column names from index spec."""
        spec = index.delta_sync_index_spec or index.direct_access_index_spec
        if not spec:
            return []

        columns = []
        if hasattr(spec, "embedding_source_columns"):
            for col in spec.embedding_source_columns or []:
                if col.name:
                    columns.append(col.name)
        return columns

    def _extract_filter_columns(self, index) -> list[dict]:
        """Extract filterable columns from index spec."""
        # Note: In production, this would query the Delta table schema
        # For now, return columns from the index spec
        return []

    def _get_supported_query_types(self, index) -> list[str]:
        """Determine supported query types for index."""
        types = ["ANN"]  # Always supported

        # Check for text columns (required for hybrid)
        spec = index.delta_sync_index_spec or index.direct_access_index_spec
        if spec and hasattr(spec, "embedding_source_columns"):
            types.append("HYBRID")

        return types

    def _build_response(
        self,
        sources: list[DiscoveredSource],
        cached: bool,
        cached_at: datetime,
        errors: list[dict] | None = None,
    ) -> DiscoveryResponse:
        """Build discovery response with grouping by type."""
        by_type = {}
        for source in sources:
            if source.source_type not in by_type:
                by_type[source.source_type] = []
            by_type[source.source_type].append(source)

        return DiscoveryResponse(
            sources=sources,
            total_count=len(sources),
            by_type=by_type,
            discovered_at=cached_at,
            cached=cached,
            cache_expires_at=cached_at + self._cache_ttl,
            errors=errors,
        )
```

### 1.3 Create API Endpoints

Add to `src/deep_research/api/v1/discovery.py`:

```python
"""Discovery API endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from deep_research.api.v1.utils.authorization import get_current_user
from deep_research.schemas.discovery import DiscoveryResponse, SourceMetadataResponse
from deep_research.services.discovery_service import DiscoveryService

router = APIRouter(prefix="/discovery", tags=["Discovery"])

# Singleton service instance
_discovery_service = DiscoveryService()


@router.get("/sources", response_model=DiscoveryResponse)
async def discover_sources(
    source_type: str | None = None,
    refresh: bool = False,
    current_user: dict = Depends(get_current_user),
) -> DiscoveryResponse:
    """Discover all available data sources for the current user."""
    return await _discovery_service.discover_all(
        user_id=current_user["user_id"],
        force_refresh=refresh,
    )


@router.get("/sources/{source_id}/metadata", response_model=SourceMetadataResponse)
async def get_source_metadata(
    source_id: str,
    current_user: dict = Depends(get_current_user),
) -> SourceMetadataResponse:
    """Get detailed metadata for a specific source."""
    response = await _discovery_service.discover_all(
        user_id=current_user["user_id"],
    )

    for source in response.sources:
        if source.source_id == source_id:
            return SourceMetadataResponse(source=source)

    raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")


@router.post("/refresh", response_model=DiscoveryResponse)
async def refresh_discovery(
    current_user: dict = Depends(get_current_user),
) -> DiscoveryResponse:
    """Force refresh the discovery cache."""
    return await _discovery_service.discover_all(
        user_id=current_user["user_id"],
        force_refresh=True,
    )
```

---

## 2. Frontend Implementation

### 2.1 Create Discovery Hook

Add to `frontend/src/hooks/useDiscovery.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export interface DiscoveredSource {
  source_id: string;
  source_type: 'vector_search' | 'genie' | 'knowledge_assistant';
  name: string;
  endpoint_name: string;
  description: string | null;
  status: 'ready' | 'syncing' | 'unavailable' | 'error';
  capabilities: string[];
  metadata: Record<string, any>;
  discovered_at: string;
}

export interface DiscoveryResponse {
  sources: DiscoveredSource[];
  total_count: number;
  by_type: Record<string, DiscoveredSource[]>;
  discovered_at: string;
  cached: boolean;
  cache_expires_at: string | null;
  errors: Array<{
    source_type: string;
    error_message: string;
  }> | null;
}

export function useDiscovery() {
  return useQuery<DiscoveryResponse>({
    queryKey: ['discovery', 'sources'],
    queryFn: async () => {
      const response = await api.get('/api/v1/discovery/sources');
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });
}

export function useRefreshDiscovery() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/v1/discovery/refresh');
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['discovery', 'sources'], data);
    },
  });
}
```

### 2.2 Create Data Source Selector Component

Add to `frontend/src/components/sources/DataSourceSelector.tsx`:

```tsx
import { useState } from 'react';
import { useDiscovery, useRefreshDiscovery, DiscoveredSource } from '@/hooks/useDiscovery';
import { Loader2, RefreshCw, Search, ChevronDown, ChevronRight } from 'lucide-react';

interface DataSourceSelectorProps {
  selectedSources: string[];
  onSelectionChange: (sourceIds: string[]) => void;
}

export function DataSourceSelector({ selectedSources, onSelectionChange }: DataSourceSelectorProps) {
  const { data, isLoading, error } = useDiscovery();
  const refreshMutation = useRefreshDiscovery();
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set(['vector_search', 'genie']));

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-4">
        <Loader2 className="h-5 w-5 animate-spin text-gray-500" />
        <span className="ml-2 text-sm text-gray-500">Discovering data sources...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-sm text-red-600">
        Failed to discover data sources. Please try again.
      </div>
    );
  }

  const toggleType = (type: string) => {
    const newExpanded = new Set(expandedTypes);
    if (newExpanded.has(type)) {
      newExpanded.delete(type);
    } else {
      newExpanded.add(type);
    }
    setExpandedTypes(newExpanded);
  };

  const toggleSource = (sourceId: string) => {
    if (selectedSources.includes(sourceId)) {
      onSelectionChange(selectedSources.filter(id => id !== sourceId));
    } else {
      onSelectionChange([...selectedSources, sourceId]);
    }
  };

  const filteredSources = data?.sources.filter(source =>
    source.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    source.description?.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const typeLabels: Record<string, string> = {
    vector_search: 'Vector Search Indexes',
    genie: 'Genie Spaces',
    knowledge_assistant: 'Knowledge Assistants',
  };

  const typeIcons: Record<string, string> = {
    vector_search: '🔍',
    genie: '🧞',
    knowledge_assistant: '🤖',
  };

  return (
    <div className="border rounded-lg bg-white">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b">
        <span className="text-sm font-medium">Data Sources ({data?.total_count || 0})</span>
        <button
          onClick={() => refreshMutation.mutate()}
          disabled={refreshMutation.isPending}
          className="p-1 hover:bg-gray-100 rounded"
        >
          <RefreshCw className={`h-4 w-4 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Search */}
      <div className="p-2 border-b">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search sources..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-2 text-sm border rounded"
          />
        </div>
      </div>

      {/* Source Groups */}
      <div className="max-h-64 overflow-y-auto">
        {Object.entries(data?.by_type || {}).map(([type, sources]) => {
          const filteredTypeSources = sources.filter(s =>
            s.name.toLowerCase().includes(searchQuery.toLowerCase())
          );

          if (filteredTypeSources.length === 0) return null;

          return (
            <div key={type}>
              <button
                onClick={() => toggleType(type)}
                className="flex items-center w-full px-3 py-2 text-sm font-medium bg-gray-50 hover:bg-gray-100"
              >
                {expandedTypes.has(type) ? <ChevronDown className="h-4 w-4 mr-1" /> : <ChevronRight className="h-4 w-4 mr-1" />}
                <span className="mr-2">{typeIcons[type]}</span>
                {typeLabels[type]} ({filteredTypeSources.length})
              </button>

              {expandedTypes.has(type) && (
                <div className="pl-4">
                  {filteredTypeSources.map(source => (
                    <SourceItem
                      key={source.source_id}
                      source={source}
                      selected={selectedSources.includes(source.source_id)}
                      onToggle={() => toggleSource(source.source_id)}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Errors */}
      {data?.errors && data.errors.length > 0 && (
        <div className="p-2 border-t bg-yellow-50">
          <span className="text-xs text-yellow-700">
            Some sources could not be discovered
          </span>
        </div>
      )}
    </div>
  );
}

function SourceItem({ source, selected, onToggle }: { source: DiscoveredSource; selected: boolean; onToggle: () => void }) {
  const statusColors: Record<string, string> = {
    ready: 'bg-green-500',
    syncing: 'bg-yellow-500',
    unavailable: 'bg-gray-400',
    error: 'bg-red-500',
  };

  return (
    <label className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer">
      <input
        type="checkbox"
        checked={selected}
        onChange={onToggle}
        disabled={source.status === 'unavailable'}
        className="mr-3"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center">
          <span className="text-sm truncate">{source.name}</span>
          <span className={`ml-2 w-2 h-2 rounded-full ${statusColors[source.status]}`} />
        </div>
        {source.description && (
          <span className="text-xs text-gray-500 truncate block">{source.description}</span>
        )}
      </div>
      <div className="flex gap-1 ml-2">
        {source.capabilities.slice(0, 2).map(cap => (
          <span key={cap} className="text-xs px-1 py-0.5 bg-gray-100 rounded">
            {cap}
          </span>
        ))}
      </div>
    </label>
  );
}
```

---

## 3. Testing

### 3.1 Unit Test

Create `tests/unit/services/test_discovery_service.py`:

```python
"""Unit tests for discovery service."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from deep_research.services.discovery_service import DiscoveryService


@pytest.fixture
def discovery_service():
    return DiscoveryService()


@pytest.mark.asyncio
async def test_discover_all_returns_empty_on_no_sources(discovery_service):
    """Test discovery returns empty list when no sources exist."""
    with patch.object(discovery_service, '_discover_vector_search', return_value=[]):
        with patch.object(discovery_service, '_discover_genie', return_value=[]):
            with patch.object(discovery_service, '_discover_serving', return_value=[]):
                response = await discovery_service.discover_all(user_id="test-user")

                assert response.total_count == 0
                assert response.sources == []
                assert response.cached is False


@pytest.mark.asyncio
async def test_discover_all_caches_results(discovery_service):
    """Test that discovery results are cached."""
    mock_sources = [MagicMock(source_id="vs:test")]

    with patch.object(discovery_service, '_discover_vector_search', return_value=mock_sources):
        with patch.object(discovery_service, '_discover_genie', return_value=[]):
            with patch.object(discovery_service, '_discover_serving', return_value=[]):
                # First call - not cached
                response1 = await discovery_service.discover_all(user_id="test-user")
                assert response1.cached is False

                # Second call - should be cached
                response2 = await discovery_service.discover_all(user_id="test-user")
                assert response2.cached is True
```

### 3.2 Integration Test

Create `tests/integration/services/test_discovery_integration.py`:

```python
"""Integration tests for discovery service with real Databricks API."""

import pytest
from deep_research.services.discovery_service import DiscoveryService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_discover_vector_search_indexes():
    """Test discovering real Vector Search indexes."""
    service = DiscoveryService()
    response = await service.discover_all(user_id="test-user")

    # Should not raise
    assert response is not None
    assert isinstance(response.sources, list)

    # Check Vector Search sources have correct metadata
    vs_sources = [s for s in response.sources if s.source_type == "vector_search"]
    for source in vs_sources:
        assert source.source_id.startswith("vs:")
        assert "ANN" in source.capabilities
```

---

## 4. Configuration

Add to `config/app.yaml`:

```yaml
discovery:
  enabled: true
  cache_ttl_minutes: 5
  parallel_discovery: true

  # Knowledge Assistant detection heuristics
  assistant_keywords:
    - assistant
    - expert
    - helper
    - advisor

  # Tags that indicate a Knowledge Assistant
  assistant_tags:
    - assistant
    - knowledge
    - expert
```

---

## 5. Next Steps

1. **Run discovery endpoint**: `GET /api/v1/discovery/sources`
2. **Integrate with research form**: Add `DataSourceSelector` to the research input
3. **Implement query configuration**: Add per-source settings UI
4. **Connect to tool factory**: Use discovered sources to create tools dynamically
