# /app/rag.py
import os
import pandas as pd
import asyncio
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
)
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from typing import List


from app.config import settings

rag_index = None
bm25_retriever = None

# Configure the global settings
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=settings.GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(model="models/text-embedding-004", api_key=settings.GOOGLE_API_KEY)


class AsyncBM25Retriever(BaseRetriever):
    """
    A retriever that wraps a BM25Retriever to run its synchronous retrieve method in a thread pool.
    """

    def __init__(self, bm25_retriever: BM25Retriever):
        self._bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Sync retrieve method."""
        return self._bm25_retriever.retrieve(query_bundle.query_str)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve method."""
        return await asyncio.to_thread(self._bm25_retriever.retrieve, query_bundle.query_str)


def build_index_from_paths(file_paths: List[str]):
    """
    Builds the RAG index and BM25 retriever from a list of local file paths.
    """
    global rag_index, bm25_retriever
    
    try:
        documents = []
        for path in file_paths:
            df = pd.read_csv(path)
            headers = df.columns.tolist()
            for index, row in df.iterrows():
                row_content_parts = []
                for header, value in row.items():
                    if pd.notna(value):
                        row_content_parts.append(f"{header}: {value}")
                
                row_content = ", ".join(row_content_parts)
                doc = Document(text=row_content, doc_id=f"{os.path.basename(path)}_row_{index}")
                documents.append(doc)

        if documents:
            rag_index = VectorStoreIndex.from_documents(documents)
            bm25_retriever = BM25Retriever.from_defaults(nodes=documents, similarity_top_k=5)
            print(f"Indexing complete. {len(documents)} rows from {len(file_paths)} files indexed.")
        else:
            rag_index = None
            bm25_retriever = None
            print("No documents found in the provided paths.")

    except Exception as e:
        rag_index = None
        bm25_retriever = None
        print(f"Error processing files from paths: {e}")


def clear_index():
    """Clears the in-memory index and retriever."""
    global rag_index, bm25_retriever
    rag_index = None
    bm25_retriever = None
    print("In-memory index has been cleared.")


def get_fusion_retriever():
    """
    Creates and returns a configured QueryFusionRetriever.
    This is the main retriever for the application.
    Returns None if the index is not ready.
    """
    global rag_index, bm25_retriever
    if not rag_index or not bm25_retriever:
        return None
    
    vector_retriever = rag_index.as_retriever(similarity_top_k=5)
    async_bm25_retriever = AsyncBM25Retriever(bm25_retriever)

    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, async_bm25_retriever],
        similarity_top_k=5,
        num_queries=1,  # generate 0 additional queries
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
    )
    return fusion_retriever


async def retrieve_context(query_text):
    """
    A debugging/utility function to retrieve context directly.
    The main chat flow uses get_fusion_retriever() to build a chat engine.
    """
    fusion_retriever = get_fusion_retriever()
    if not fusion_retriever:
        return "[RAG Index not ready. Please upload documents first.]"

    retrieved_nodes = await fusion_retriever.aretrieve(query_text)
    
    # --- Debugging ---
    print("\n--- RAG DEBUG START ---")
    print(f"Query: {query_text}")
    print(f"Retrieved {len(retrieved_nodes)} source nodes via Fusion Retriever.")
    for i, node in enumerate(retrieved_nodes):
        print(f"  Node {i+1} (Score: {node.score:.4f}):")
        print(f"    Content: {node.text[:200]}...") # Print first 200 chars
    print("--- RAG DEBUG END ---\n")
    # --- End Debugging ---

    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    return context_str

# --- Auto-build index on startup --- is now handled by the lifespan manager
# index_documents()