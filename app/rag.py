# /app/rag.py
import os
import pandas as pd
import asyncio
import logging
import hashlib
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
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import List, Optional, Dict

from app.config import settings
from app.chroma_client import chroma_manager
from app.multi_strategy_search import multi_strategy_search, MultiStrategySearchResult
from app.error_handler import error_handling_context, ErrorContext, ErrorCategory

# Import performance monitoring
try:
    from app.performance_monitor import performance_monitor, monitor_performance, OperationType
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    performance_monitor = None

logger = logging.getLogger(__name__)

# Legacy global variables for backward compatibility
rag_index = None
bm25_retriever = None

# User-specific indexes and retrievers
user_indexes: Dict[str, VectorStoreIndex] = {}
user_bm25_retrievers: Dict[str, BM25Retriever] = {}

# Configure the global settings
Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)
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


async def get_user_index(user_id: str) -> Optional[VectorStoreIndex]:
    """Get or create user-specific VectorStoreIndex backed by ChromaDB with enhanced validation."""
    async with error_handling_context("get_user_index", user_id=user_id) as context:
        # Enhanced logging for user context debugging
        logger.info(f"Getting user index for: {user_id}")
        logger.debug(f"Currently cached user indexes: {list(user_indexes.keys())}")
        
        # Enhanced user_id validation
        if not user_id or not isinstance(user_id, str):
            logger.error(f"Invalid user_id provided: {user_id} (type: {type(user_id)})")
            return None
        
        # Validate user_id doesn't contain problematic characters
        problematic_chars = ['\x00', '\n', '\r', '\t']
        for char in problematic_chars:
            if char in user_id:
                logger.error(f"User ID contains invalid character: {repr(char)}")
                return None
        
        # Check cached index first with validation
        if user_id in user_indexes:
            cached_index = user_indexes[user_id]
            logger.info(f"Found cached index for user: {user_id}")
            
            # Validate cached index is still functional
            try:
                # Quick validation - try to create a retriever
                test_retriever = cached_index.as_retriever(similarity_top_k=1)
                if test_retriever:
                    logger.debug(f"Cached index for {user_id} is functional")
                    return cached_index
                else:
                    logger.warning(f"Cached index for {user_id} failed retriever test, removing from cache")
                    del user_indexes[user_id]
            except Exception as validation_error:
                logger.warning(f"Cached index for {user_id} validation failed: {validation_error}, removing from cache")
                del user_indexes[user_id]
        
        try:
            # Get user's ChromaDB collection with enhanced error handling
            logger.debug(f"Attempting to get ChromaDB collection for user: {user_id}")
            collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
            
            # Enhanced collection validation
            if not collection:
                logger.error(f"ChromaDB collection creation failed for user: {user_id}")
                raise Exception("Collection creation failed")
            
            # Validate collection name matches expected pattern
            expected_hash = hashlib.md5(user_id.encode('utf-8')).hexdigest()
            expected_name = f"{settings.CHROMA_COLLECTION_PREFIX}{expected_hash}"
            if collection.name != expected_name:
                logger.warning(f"Collection name mismatch for {user_id}: expected {expected_name}, got {collection.name}")
            
            # Check if collection has documents with enhanced validation
            try:
                doc_count = await asyncio.to_thread(collection.count)
                logger.info(f"ChromaDB collection for {user_id} contains {doc_count} documents")
                
                if doc_count == 0:
                    logger.warning(f"ChromaDB collection for {user_id} is empty - checking if documents should exist")
                    
                    # Check if user has documents that should be indexed
                    user_doc_dir = os.path.join("user_documents", user_id)
                    if os.path.exists(user_doc_dir):
                        csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
                        if csv_files:
                            logger.warning(f"User {user_id} has {len(csv_files)} CSV files but empty collection - may need reindexing")
                        else:
                            logger.info(f"User {user_id} has no CSV files - empty collection is expected")
                    else:
                        logger.info(f"User {user_id} document directory does not exist - empty collection is expected")
                else:
                    # Validate collection health by trying to peek at documents
                    try:
                        sample = await asyncio.to_thread(collection.peek, 1)
                        if sample and sample.get('documents'):
                            logger.debug(f"Collection for {user_id} is healthy - sample document retrieved")
                        else:
                            logger.warning(f"Collection for {user_id} count is {doc_count} but peek returned no documents")
                    except Exception as peek_error:
                        logger.warning(f"Collection for {user_id} peek failed: {peek_error} - collection may be corrupted")
                        
            except Exception as count_error:
                logger.warning(f"Could not count documents in collection for {user_id}: {count_error}")
                # Continue anyway - we'll try to create the index and see what happens
            
            # Create ChromaVectorStore with validation
            try:
                vector_store = ChromaVectorStore(chroma_collection=collection)
                if not vector_store:
                    logger.error(f"ChromaVectorStore creation failed for user: {user_id}")
                    raise Exception("Vector store creation failed")
                logger.debug(f"Successfully created ChromaVectorStore for user: {user_id}")
            except Exception as vs_error:
                logger.error(f"Failed to create ChromaVectorStore for {user_id}: {vs_error}")
                raise
            
            # Create VectorStoreIndex with ChromaVectorStore
            try:
                index = VectorStoreIndex.from_vector_store(vector_store)
                if not index:
                    logger.error(f"VectorStoreIndex creation failed for user: {user_id}")
                    raise Exception("Index creation failed")
                logger.debug(f"Successfully created VectorStoreIndex for user: {user_id}")
            except Exception as index_error:
                logger.error(f"Failed to create VectorStoreIndex for {user_id}: {index_error}")
                raise
            
            # Validate index functionality before caching
            try:
                test_retriever = index.as_retriever(similarity_top_k=1)
                if not test_retriever:
                    logger.error(f"Index for {user_id} failed retriever creation test")
                    raise Exception("Index retriever test failed")
                logger.debug(f"Index for {user_id} passed functionality test")
            except Exception as test_error:
                logger.error(f"Index functionality test failed for {user_id}: {test_error}")
                raise
            
            # Cache the validated index
            user_indexes[user_id] = index
            
            logger.info(f"Successfully created/retrieved and validated ChromaDB-backed index for user: {user_id}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB index for {user_id}: {e}")
            logger.info(f"Attempting fallback to in-memory index for user: {user_id}")
            
            # Enhanced fallback to in-memory index
            try:
                # Create empty in-memory index with proper validation
                empty_docs = []
                index = VectorStoreIndex.from_documents(empty_docs)
                
                if not index:
                    logger.error(f"Failed to create fallback in-memory index for {user_id}")
                    return None
                
                # Test the fallback index
                try:
                    test_retriever = index.as_retriever(similarity_top_k=1)
                    if not test_retriever:
                        logger.error(f"Fallback index for {user_id} failed retriever test")
                        return None
                except Exception as test_error:
                    logger.error(f"Fallback index test failed for {user_id}: {test_error}")
                    return None
                
                # Cache the fallback index
                user_indexes[user_id] = index
                logger.info(f"Created and validated fallback in-memory index for user: {user_id}")
                return index
                
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback index for {user_id}: {fallback_error}")
                return None


async def build_user_index(user_id: str, file_paths: List[str]) -> Optional[VectorStoreIndex]:
    """Build user-specific index from file paths using ChromaDB with performance monitoring."""
    # Use performance monitoring if available
    if PERFORMANCE_MONITORING_AVAILABLE and performance_monitor is not None:
        async with monitor_performance(
            performance_monitor,
            OperationType.INDEX_BUILDING,
            f"build_user_index_{user_id}",
            user_id=user_id,
            file_count=len(file_paths)
        ):
            return await _build_user_index_impl(user_id, file_paths)
    else:
        return await _build_user_index_impl(user_id, file_paths)


async def _build_user_index_impl(user_id: str, file_paths: List[str]) -> Optional[VectorStoreIndex]:
    """Implementation of build_user_index without performance monitoring wrapper."""
    try:
        # Get user's ChromaDB collection (now synchronous)
        collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
        use_chromadb = True
    except Exception as e:
        logger.error(f"Failed to get ChromaDB collection for {user_id}: {e}")
        logger.info(f"Falling back to in-memory storage for user: {user_id}")
        use_chromadb = False
    
    try:
        # Create documents from file paths
        documents = []
        for path in file_paths:
            df = pd.read_csv(path)
            # Standardize column names for consistent metadata
            df.columns = [col.strip().lower() for col in df.columns]
            
            for index, row in df.iterrows():
                # The main text content
                row_content = ", ".join([f"{header}: {value}" for header, value in row.items() if pd.notna(value)])
                
                # Create metadata dictionary for filtering
                metadata = {
                    "user_id": user_id,
                    "file_name": os.path.basename(path),
                    "row_index": index,
                    "upload_timestamp": pd.Timestamp.now().isoformat()
                }
                
                # Add all CSV columns as metadata
                for key, value in row.to_dict().items():
                    if pd.notna(value):
                        metadata[key] = str(value)
                
                doc = Document(
                    text=row_content, 
                    doc_id=f"{os.path.basename(path)}_row_{index}_{user_id}",
                    metadata=metadata
                )
                documents.append(doc)

        if documents:
            if use_chromadb:
                # Create ChromaVectorStore
                vector_store = ChromaVectorStore(chroma_collection=collection)
                
                # Create VectorStoreIndex and add documents
                index = VectorStoreIndex.from_documents(
                    documents, 
                    vector_store=vector_store,
                    embed_batch_size=100
                )
                logger.info(f"Built ChromaDB index for user {user_id} with {len(documents)} documents")
            else:
                # Fallback to in-memory index
                index = VectorStoreIndex.from_documents(documents, embed_batch_size=100)
                logger.info(f"Built fallback in-memory index for user {user_id} with {len(documents)} documents")
            
            # Cache the index
            user_indexes[user_id] = index
            
            # Create BM25 retriever for this user
            try:
                # For ChromaDB indexes, we need to create nodes from the original documents
                # since the docstore might not contain the nodes directly
                if use_chromadb:
                    # Convert documents to nodes for BM25
                    from llama_index.core.schema import TextNode
                    nodes = []
                    for doc in documents:
                        node = TextNode(
                            text=doc.text,
                            metadata=doc.metadata,
                            id_=doc.doc_id
                        )
                        nodes.append(node)
                    logger.info(f"Created {len(nodes)} nodes from documents for BM25 retriever")
                else:
                    # For in-memory indexes, get nodes from docstore
                    nodes = list(index.docstore.docs.values()) if hasattr(index.docstore, 'docs') else []
                
                if nodes:
                    user_bm25_retrievers[user_id] = BM25Retriever.from_defaults(
                        nodes=nodes, 
                        similarity_top_k=5
                    )
                    logger.info(f"Successfully created BM25 retriever for user {user_id} with {len(nodes)} nodes")
                else:
                    logger.warning(f"No nodes available for BM25 retriever for user {user_id}")
            except Exception as bm25_error:
                logger.error(f"Failed to create BM25 retriever for user {user_id}: {bm25_error}")
                import traceback
                traceback.print_exc()
                # Continue without BM25 - the system can still work with just vector search
            
            return index
        else:
            logger.warning(f"No documents found for user {user_id}")
            return None

    except Exception as e:
        logger.error(f"Error building user index for {user_id}: {e}")
        return None


async def clear_user_index(user_id: str) -> bool:
    """Clear user-specific index and ChromaDB collection."""
    try:
        # Remove from local cache
        if user_id in user_indexes:
            del user_indexes[user_id]
        if user_id in user_bm25_retrievers:
            del user_bm25_retrievers[user_id]
        
        # Try to delete ChromaDB collection (now synchronous)
        try:
            success = await asyncio.to_thread(chroma_manager.delete_user_collection, user_id)
            if success:
                logger.info(f"Cleared ChromaDB collection for user: {user_id}")
            else:
                logger.warning(f"Failed to clear ChromaDB collection for user: {user_id}")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection for {user_id}: {e}")
            # Continue anyway, as we've cleared the local cache
            success = True
        
        logger.info(f"Cleared local index cache for user: {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing user index for {user_id}: {e}")
        return False


def build_index_from_paths(file_paths: List[str]):
    """
    Legacy function for backward compatibility.
    Now uses the first available user or creates a default user.
    """
    global rag_index, bm25_retriever
    
    # For backward compatibility, use a default user
    default_user = "default_user"
    
    try:
        # Run async function in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use asyncio.run
            # This is a limitation of the legacy interface
            logger.warning("build_index_from_paths called from async context - using cached index if available")
            rag_index = user_indexes.get(default_user)
            bm25_retriever = user_bm25_retrievers.get(default_user)
        else:
            # We're in a sync context, can use asyncio.run
            rag_index = asyncio.run(build_user_index(default_user, file_paths))
            bm25_retriever = user_bm25_retrievers.get(default_user)
        
        if rag_index:
            logger.info(f"Legacy build_index_from_paths completed with {len(file_paths)} files")
        else:
            logger.error("Legacy build_index_from_paths failed")
            
    except Exception as e:
        logger.error(f"Error in legacy build_index_from_paths: {e}")
        rag_index = None
        bm25_retriever = None


def clear_index():
    """Legacy function for backward compatibility."""
    global rag_index, bm25_retriever
    
    # Clear legacy global variables
    rag_index = None
    bm25_retriever = None
    
    # Clear all user indexes
    user_indexes.clear()
    user_bm25_retrievers.clear()
    
    logger.info("Cleared all indexes (legacy function)")


def get_fusion_retriever(user_id: Optional[str] = None):
    """
    Creates and returns a configured QueryFusionRetriever.
    Now supports user-specific retrievers with enhanced validation.
    """
    if user_id:
        # Use user-specific retrievers with comprehensive validation
        logger.info(f"Getting fusion retriever for user: {user_id}")
        
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            logger.error(f"Invalid user_id provided to get_fusion_retriever: {user_id}")
            return None
        
        user_index = user_indexes.get(user_id)
        user_bm25 = user_bm25_retrievers.get(user_id)
        
        # Enhanced debugging with detailed context
        logger.debug(f"Available user indexes: {list(user_indexes.keys())}")
        logger.debug(f"Available user BM25 retrievers: {list(user_bm25_retrievers.keys())}")
        logger.info(f"User index found for {user_id}: {user_index is not None}")
        logger.info(f"User BM25 found for {user_id}: {user_bm25 is not None}")
        
        # Detailed validation of components
        if not user_index:
            logger.warning(f"Cannot create fusion retriever for {user_id}: user index not available")
            logger.info(f"Suggestion: Call build_user_index() or get_user_index() first for user {user_id}")
            return None
            
        if not user_bm25:
            logger.warning(f"Cannot create fusion retriever for {user_id}: BM25 retriever not available")
            logger.info(f"Suggestion: BM25 retriever should be created during index building for user {user_id}")
            return None
        
        # Validate that index can create a retriever
        try:
            vector_retriever = user_index.as_retriever(similarity_top_k=5)
            if not vector_retriever:
                logger.error(f"Failed to create vector retriever from index for user {user_id}")
                return None
            logger.debug(f"Successfully created vector retriever for user {user_id}")
        except Exception as e:
            logger.error(f"Error creating vector retriever for user {user_id}: {e}")
            return None
        
        # Validate BM25 retriever
        try:
            async_bm25_retriever = AsyncBM25Retriever(user_bm25)
            if not async_bm25_retriever:
                logger.error(f"Failed to create async BM25 retriever for user {user_id}")
                return None
            logger.debug(f"Successfully created async BM25 retriever for user {user_id}")
        except Exception as e:
            logger.error(f"Error creating async BM25 retriever for user {user_id}: {e}")
            return None
        
        logger.info(f"Successfully created individual retrievers for user: {user_id}")
        
    else:
        # Legacy mode - use global variables
        global rag_index, bm25_retriever
        logger.info(f"Using legacy mode - rag_index: {rag_index is not None}, bm25_retriever: {bm25_retriever is not None}")
        
        if not rag_index or not bm25_retriever:
            logger.warning("Cannot create fusion retriever in legacy mode: missing components")
            return None
        
        try:
            vector_retriever = rag_index.as_retriever(similarity_top_k=5)
            async_bm25_retriever = AsyncBM25Retriever(bm25_retriever)
        except Exception as e:
            logger.error(f"Error creating retrievers in legacy mode: {e}")
            return None

    # Create fusion retriever with error handling
    try:
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, async_bm25_retriever],
            similarity_top_k=5,
            num_queries=1,  # generate 0 additional queries
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
        )
        
        if not fusion_retriever:
            logger.error(f"QueryFusionRetriever creation returned None for user: {user_id}")
            return None
            
        logger.info(f"Successfully created fusion retriever for user: {user_id}")
        return fusion_retriever
        
    except Exception as e:
        logger.error(f"Error creating QueryFusionRetriever for user {user_id}: {e}")
        return None


async def retrieve_context(query_text: str, user_id: Optional[str] = None):
    """
    A debugging/utility function to retrieve context directly.
    Now supports user-specific context retrieval.
    """
    fusion_retriever = get_fusion_retriever(user_id)
    if not fusion_retriever:
        return "[RAG Index not ready. Please upload documents first.]"

    retrieved_nodes = await fusion_retriever.aretrieve(query_text)
    
    # Debugging
    logger.debug(f"Query: {query_text}")
    logger.debug(f"Retrieved {len(retrieved_nodes)} source nodes via Fusion Retriever for user: {user_id}")
    for i, node in enumerate(retrieved_nodes):
        logger.debug(f"Node {i+1} (Score: {node.score:.4f}): {node.text[:200]}...")

    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    return context_str


async def retrieve_context_optimized(query_text: str, user_id: Optional[str] = None) -> MultiStrategySearchResult:
    """
    Enhanced context retrieval using multi-strategy search logic with performance monitoring.
    
    This function implements the multi-strategy search approach that:
    1. Tries multiple search strategies (exact, address-only, fuzzy, partial)
    2. Preserves exact address formatting from queries
    3. Implements fallback strategies when primary search fails
    4. Ranks and validates search results
    
    Args:
        query_text: The user's search query
        user_id: Optional user ID for user-specific search
        
    Returns:
        MultiStrategySearchResult containing the best results from all strategies
    """
    # Use performance monitoring if available
    if PERFORMANCE_MONITORING_AVAILABLE and performance_monitor is not None:
        async with monitor_performance(
            performance_monitor,
            OperationType.MULTI_STRATEGY_SEARCH,
            f"retrieve_context_optimized_{user_id}",
            user_id=user_id,
            query_length=len(query_text)
        ):
            return await _retrieve_context_optimized_impl(query_text, user_id)
    else:
        return await _retrieve_context_optimized_impl(query_text, user_id)


async def _retrieve_context_optimized_impl(query_text: str, user_id: Optional[str] = None) -> MultiStrategySearchResult:
    """Implementation of retrieve_context_optimized without performance monitoring wrapper."""
    fusion_retriever = get_fusion_retriever(user_id)
    if not fusion_retriever:
        logger.warning(f"No fusion retriever available for user: {user_id}")
        # Return empty result
        return MultiStrategySearchResult(
            original_query=query_text,
            best_result=None,
            all_results=[],
            total_execution_time_ms=0.0,
            nodes_found=[]
        )
    
    logger.info(f"Starting multi-strategy search for user '{user_id}' with query: '{query_text}'")
    
    try:
        # Perform multi-strategy search
        search_result = await multi_strategy_search(fusion_retriever, query_text)
        
        logger.info(f"Multi-strategy search completed for user '{user_id}': "
                   f"{len(search_result.nodes_found)} nodes found from {len(search_result.all_results)} strategies "
                   f"in {search_result.total_execution_time_ms:.2f}ms")
        
        # Record individual strategy performance if monitoring is available
        if PERFORMANCE_MONITORING_AVAILABLE and performance_monitor is not None:
            for result in search_result.all_results:
                performance_monitor.record_metric(
                    operation_type=OperationType.SEARCH_OPERATION,
                    operation_name=f"search_strategy_{result.strategy}",
                    duration_ms=result.execution_time_ms,
                    success=result.success,
                    user_id=user_id,
                    strategy=result.strategy,
                    nodes_found=len(result.nodes),
                    query_used=result.query_used
                )
        
        # Log details about each strategy's performance
        for result in search_result.all_results:
            logger.debug(f"Strategy '{result.strategy}': {len(result.nodes)} nodes, "
                        f"success={result.success}, time={result.execution_time_ms:.2f}ms, "
                        f"query='{result.query_used}'")
        
        if search_result.best_result:
            logger.info(f"Best strategy selected: '{search_result.best_result.strategy}' "
                       f"with {len(search_result.best_result.nodes)} nodes")
        
        return search_result
        
    except Exception as e:
        logger.error(f"Error during multi-strategy search for user '{user_id}': {e}")
        # Return empty result on error
        return MultiStrategySearchResult(
            original_query=query_text,
            best_result=None,
            all_results=[],
            total_execution_time_ms=0.0,
            nodes_found=[]
        )


async def retrieve_context_with_fallback(query_text: str, user_id: Optional[str] = None) -> str:
    """
    Retrieve context using multi-strategy search with fallback to original method.
    
    This function provides backward compatibility while using the enhanced search logic.
    It returns a formatted context string like the original retrieve_context function.
    
    Args:
        query_text: The user's search query
        user_id: Optional user ID for user-specific search
        
    Returns:
        Formatted context string from the best search results
    """
    # Try multi-strategy search first
    multi_result = await retrieve_context_optimized(query_text, user_id)
    
    if multi_result.nodes_found:
        # Format the context from multi-strategy results
        logger.info(f"Using multi-strategy search results: {len(multi_result.nodes_found)} nodes")
        context_str = "\n\n".join([node.get_content() for node in multi_result.nodes_found])
        return context_str
    else:
        # Fallback to original method
        logger.info("Multi-strategy search found no results, falling back to original method")
        return await retrieve_context(query_text, user_id)