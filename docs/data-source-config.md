# Data Source Configuration Guide

This guide explains how to configure enterprise data sources in the Deep Research Agent, including Vector Search indexes, Genie spaces, and Knowledge Assistants.

## Overview

The Deep Research Agent supports three types of enterprise data sources:

| Source Type | Description | Use Case |
|-------------|-------------|----------|
| **Vector Search** | Databricks Vector Search indexes | Semantic search over documents, FAQs, knowledge bases |
| **Genie** | Databricks Genie (AI/BI) spaces | Natural language queries against relational data |
| **Knowledge Assistant** | Model Serving endpoints | Domain expert Q&A, specialized models |

All sources use **OBO (On-Behalf-Of) authentication** - the system queries data using your identity, so you only see sources you have permission to access.

## Auto-Discovery

The system automatically discovers available data sources when you first access the research interface.

### What Gets Discovered

1. **Vector Search Indexes**: All indexes you have access to across all endpoints
2. **Genie Spaces**: All AI/BI spaces you can query
3. **Knowledge Assistants**: Serving endpoints identified as Q&A assistants

### Discovery Cache

- Results are cached for **5 minutes** per user
- Click the **Refresh** button to force re-discovery
- Cache helps reduce API calls and improves responsiveness

### Source Status Indicators

| Status | Meaning |
|--------|---------|
| **Ready** | Source is online and accessible |
| **Syncing** | Index is currently synchronizing |
| **Unavailable** | Source is offline or inaccessible |
| **Error** | Failed to connect or authenticate |

## Configuring Vector Search Sources

Vector Search sources support the most configuration options.

### Query Types

| Type | Description | When to Use |
|------|-------------|-------------|
| **ANN** | Approximate Nearest Neighbors | Default; fast semantic search |
| **HYBRID** | ANN + keyword matching | Better recall for specific terms |
| **FULL_TEXT** | Keyword search only | When semantic matching isn't needed |

Not all indexes support all query types. The UI disables unsupported options.

### Query Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| **Num Results** | 1-100 | 10 | Number of results to return |
| **Score Threshold** | 0.0-1.0 | None | Minimum similarity score (optional) |
| **Columns** | List | All | Specific columns to return |
| **Reranking** | On/Off | Off | Enable model-based reranking |

### Configuring Filters

Filters let you narrow search results by metadata columns.

#### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `equals` | Exact match | `category = 'docs'` |
| `not equals` | Exclude value | `status != 'archived'` |
| `less than` | Numeric/date comparison | `date < '2024-01-01'` |
| `less or equal` | Numeric/date comparison | `score <= 100` |
| `greater than` | Numeric/date comparison | `count > 0` |
| `greater or equal` | Numeric/date comparison | `price >= 10.0` |
| `contains` | Substring match (LIKE) | `title contains 'python'` |
| `not contains` | Exclude substring | `content not contains 'draft'` |
| `in list` | Match any value in list | `id in (1, 2, 3)` |

#### Filter Limits

- Maximum **1,024 values** in an IN clause
- Maximum **10 filters** per configuration
- Filters are combined with AND logic

#### Example Configuration

```
Source: product_docs
Query Type: HYBRID
Num Results: 20
Score Threshold: 0.7
Filters:
  - category = 'documentation'
  - date > '2024-01-01'
  - status != 'archived'
```

### Reranking

When enabled, results are re-scored using a cross-encoder model for higher relevance.

1. Enable the **Reranking** toggle
2. Select columns to use for reranking (typically `content` or `text`)
3. The top results will be reranked after initial retrieval

Requirements:
- `databricks-vectorsearch >= 0.57`
- Index must support reranking
- Increases latency slightly but improves relevance

## Configuring Genie Sources

Genie spaces require minimal configuration.

### Settings

| Setting | Description |
|---------|-------------|
| **Space ID** | The Genie space identifier |
| **Description** | Optional description for your reference |
| **Example Questions** | Sample questions that work well |

### Follow-up Queries

Genie maintains conversation context. The researcher can:
- Ask follow-up questions referencing previous results
- Drill down into specific aspects of the data
- Build on previous SQL queries

### Result Handling

- Large result sets are truncated to **100 rows** by default
- Generated SQL is shown for transparency
- The agent generates a narrative summary of tabular results

## Configuring Knowledge Assistants

Knowledge Assistants are specialized Q&A endpoints.

### Settings

| Setting | Description |
|---------|-------------|
| **Endpoint Name** | The serving endpoint name |
| **Description** | What this assistant specializes in |
| **Pass Context** | Send research context with questions |

### Context Passing

When **Pass Context** is enabled:
- The assistant receives relevant research context
- Enables more informed, contextual answers
- May increase token usage

## Adding Custom Data Sources

Beyond auto-discovered sources, you can add custom connections.

### Adding a Vector Search Source

1. Navigate to **Settings > Data Sources**
2. Click **Add Source**
3. Select **Vector Search**
4. Enter the fully-qualified index name: `catalog.schema.index_name`
5. The system validates your access via OBO
6. Configure query settings (optional)
7. Save

### Adding a Genie Source

1. Navigate to **Settings > Data Sources**
2. Click **Add Source**
3. Select **Genie**
4. Enter the Space ID (from Genie URL)
5. Add description and example questions
6. Save

### Adding a Knowledge Assistant

1. Navigate to **Settings > Data Sources**
2. Click **Add Source**
3. Select **Knowledge Assistant**
4. Enter the endpoint name
5. Add description
6. Configure context passing
7. Save

## Source Visibility

Each source has a visibility setting:

| Visibility | Who Can See |
|------------|-------------|
| **Private** | Only you |
| **Workspace** | All workspace users |

Workspace-visible sources appear in the discovery list for other users (if they have underlying access).

## Source Scope in Research

When starting research, you can control which source categories to use.

### Scope Options

| Scope | Description |
|-------|-------------|
| **All** | Use both web and enterprise sources |
| **Enterprise Only** | Only use configured data sources |
| **Web Only** | Only use web search |

### Per-Source Selection

Expand the scope selector to enable/disable individual sources for a research session.

## Best Practices

### Performance

1. **Set appropriate result limits** - Start with 10-20 results, increase if needed
2. **Use filters** - Narrow results to relevant content
3. **Enable HYBRID** - Better recall with minimal overhead
4. **Consider reranking** - For critical searches where quality matters

### Accuracy

1. **Set score thresholds** - Filter out low-relevance results
2. **Use specific filters** - Date ranges, categories, etc.
3. **Combine sources** - Cross-reference Vector Search with Genie

### Security

1. **OBO authentication** - All queries use your identity
2. **Access is validated** - You only see sources you can access
3. **Results are filtered** - Unity Catalog permissions apply

## Troubleshooting

### Source Not Discovered

- Verify you have access to the underlying resource
- Check if the index/space/endpoint is online
- Click Refresh to re-discover
- Verify OBO is configured correctly

### Validation Failed

- Your OBO token may have expired (re-authenticate)
- The source may have been deleted or access revoked
- Network connectivity issues

### Query Errors

- Check filter column names match the index schema
- Verify query type is supported by the index
- Review score threshold (too high may return no results)

### Slow Queries

- Reduce `num_results` to fetch fewer documents
- Add filters to narrow the search space
- Consider whether reranking is necessary
- Check if the underlying index needs optimization

## Related Documentation

- [API Reference](./api.md#enterprise-data-sources-api) - REST API details
- [Architecture](./architecture.md) - System overview
- [Plugin Development](./plugin-development.md) - Extend with custom sources
