"""
Databricks Vector Search integration for internal knowledge retrieval.
"""

import os
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:
    from databricks_langchain import VectorSearchRetrieverTool, DatabricksVectorSearch
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    
    # Fallback implementation for development
    class VectorSearchRetrieverTool:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_relevant_documents(self, query: str) -> List[Document]:
            return []


class DatabricksVectorSearchRetriever(BaseRetriever):
    """Custom retriever for Databricks Vector Search."""
    
    def __init__(self, index_name: str, k: int = 5, **kwargs):
        """Initialize Vector Search retriever."""
        super().__init__(**kwargs)
        self.index_name = index_name
        self.k = k
        
        if DATABRICKS_AVAILABLE:
            try:
                # Initialize Databricks Vector Search
                self.vector_search = DatabricksVectorSearch(
                    index_name=index_name,
                    text_column="content",
                    columns=["source", "title", "url"]
                )
            except Exception as e:
                print(f"Warning: Could not initialize Vector Search: {e}")
                self.vector_search = None
        else:
            self.vector_search = None
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from Vector Search index."""
        if not self.vector_search:
            return []
        
        try:
            results = self.vector_search.similarity_search(query, k=self.k)
            
            # Convert to Document format
            documents = []
            for result in results:
                if hasattr(result, 'page_content'):
                    # Already a Document
                    documents.append(result)
                else:
                    # Convert dict to Document
                    content = result.get("content", str(result))
                    metadata = {
                        "source": result.get("source", "vector_search"),
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0.0)
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
            
        except Exception as e:
            print(f"Vector Search error: {e}")
            return []


def setup_vector_search_retriever(index_name: Optional[str] = None, k: int = 5) -> Optional[DatabricksVectorSearchRetriever]:
    """
    Set up Vector Search retriever for internal knowledge base.
    
    Args:
        index_name: Databricks Vector Search index name (e.g., "main.default.docs_index")
        k: Number of documents to retrieve
        
    Returns:
        Configured retriever or None if not available
    """
    if not index_name:
        return None
    
    if not DATABRICKS_AVAILABLE:
        print("Warning: databricks_langchain not available, Vector Search disabled")
        return None
    
    try:
        return DatabricksVectorSearchRetriever(index_name=index_name, k=k)
    except Exception as e:
        print(f"Warning: Could not set up Vector Search: {e}")
        return None


def create_vector_search_tool(index_name: str, k: int = 5) -> Optional[VectorSearchRetrieverTool]:
    """
    Create a Vector Search tool for LangChain agents.
    
    Args:
        index_name: Databricks Vector Search index name
        k: Number of documents to retrieve
        
    Returns:
        Configured tool or None if not available
    """
    retriever = setup_vector_search_retriever(index_name, k)
    
    if not retriever:
        return None
    
    if DATABRICKS_AVAILABLE:
        try:
            return VectorSearchRetrieverTool(retriever=retriever)
        except Exception as e:
            print(f"Warning: Could not create Vector Search tool: {e}")
            return None
    
    return None