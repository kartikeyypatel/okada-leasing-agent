# /app/rag.py
import os
import pandas as pd
import asyncio
import logging
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
    StorageContext,
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
import json
from PyPDF2 import PdfReader
import chromadb

from app.config import settings

logger = logging.getLogger(__name__)


def sanitize_collection_name(user_id: str) -> str:
    """
    Sanitize user ID to create a valid Chroma collection name.
    Chroma requires names with 3-512 characters from [a-zA-Z0-9._-], 
    starting and ending with [a-zA-Z0-9].
    """
    import re
    import hashlib
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', user_id)
    
    # Ensure it starts and ends with alphanumeric
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    
    # If the sanitized name is too short or empty, use a hash
    if len(sanitized) < 3:
        hash_obj = hashlib.md5(user_id.encode())
        sanitized = f"user_{hash_obj.hexdigest()[:8]}"
    
    # Ensure it's not too long (max 512 chars, but we'll keep it reasonable)
    if len(sanitized) > 50:
        hash_obj = hashlib.md5(user_id.encode())
        sanitized = f"{sanitized[:40]}_{hash_obj.hexdigest()[:8]}"
    
    return sanitized


def sanitize_collection_name(user_id: str) -> str:
    """
    Sanitize user ID to create a valid Chroma collection name.
    Chroma requires names with 3-512 characters from [a-zA-Z0-9._-], 
    starting and ending with [a-zA-Z0-9].
    """
    import re
    import hashlib
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', user_id)
    
    # Ensure it starts and ends with alphanumeric
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    
    # If the sanitized name is too short or empty, use a hash
    if len(sanitized) < 3:
        hash_obj = hashlib.md5(user_id.encode())
        sanitized = f"user_{hash_obj.hexdigest()[:8]}"
    
    # Ensure it's not too long (max 512 chars, but we'll keep it reasonable)
    if len(sanitized) > 50:
        hash_obj = hashlib.md5(user_id.encode())
        sanitized = f"{sanitized[:40]}_{hash_obj.hexdigest()[:8]}"
    
    return sanitized


async def create_enhanced_csv_embeddings(df: pd.DataFrame, file_path: str, user_id: str) -> List[Document]:
    """
    Create enhanced text embeddings for CSV data by converting rows to descriptive text.
    This improves RAG performance by making CSV data more searchable and contextual.
    """
    documents = []
    file_name = os.path.basename(file_path)
    
    # Analyze the CSV structure to create better text representations
    columns = df.columns.tolist()
    
    # Detect if this looks like a property listing CSV based on common column names
    property_indicators = ['address', 'rent', 'size', 'floor', 'suite', 'building', 'price', 'sqft', 'square']
    is_property_data = any(any(indicator in col.lower() for indicator in property_indicators) for col in columns)
    
    logger.info(f"Processing CSV {file_name} with {len(df)} rows and {len(columns)} columns")
    logger.info(f"Detected as property data: {is_property_data}")
    
    for index, row in df.iterrows():
        try:
            # Create multiple text representations for better embedding
            
            # 1. Structured description format
            structured_text = create_structured_description(row, columns, is_property_data)
            
            # 2. Natural language description
            natural_text = create_natural_language_description(row, columns, is_property_data)
            
            # 3. JSON format for exact matching
            json_text = create_json_representation(row)
            
            # 4. Key-value pairs for specific searches
            kv_text = create_key_value_representation(row)
            
            # Create more concise text for faster embedding (performance optimization)
            combined_text = f"{structured_text}\n{natural_text}".strip()
            
            # Create simplified metadata (fix Chroma metadata error)
            metadata = {
                "user_id": str(user_id),
                "file_name": str(file_name),
                "row_index": int(index),
                "data_type": "property" if is_property_data else "general"
            }
            
            # Add only essential metadata as strings/numbers (Chroma requirement)
            essential_fields = ['address', 'rent', 'size', 'floor', 'suite']
            for field in essential_fields:
                if field in row.index and pd.notna(row[field]):
                    clean_key = str(field).lower().replace(" ", "_").replace("-", "_")
                    value = row[field]
                    # Ensure metadata values are str, int, float, or None
                    if isinstance(value, (str, int, float)):
                        metadata[clean_key] = value
                    else:
                        metadata[clean_key] = str(value)
            
            # Create the document
            doc = Document(
                text=combined_text,
                doc_id=f"{file_name}_row_{index}_{user_id}",
                metadata=metadata
            )
            documents.append(doc)
            
            # Skip focused documents completely for maximum performance
            # Focused documents are disabled to optimize retrieval speed
                
        except Exception as e:
            logger.error(f"Error processing row {index} in {file_name}: {e}")
            continue
    
    logger.info(f"Created {len(documents)} enhanced embedding documents from {file_name}")
    return documents


def create_structured_description(row: pd.Series, columns: List[str], is_property_data: bool) -> str:
    """Create a structured description of the row data."""
    if is_property_data:
        return create_property_structured_description(row)
    else:
        return create_general_structured_description(row, columns)


def create_property_structured_description(row: pd.Series) -> str:
    """Create a structured description specifically for property data."""
    parts = []
    
    # Address/Location
    address_fields = ['address', 'building_address', 'location', 'building_name']
    for field in address_fields:
        if field in row.index and pd.notna(row[field]):
            parts.append(f"Property Address: {row[field]}")
            break
    
    # Floor and Suite
    floor_fields = ['floor', 'floor_number', 'level']
    suite_fields = ['suite', 'suite_number', 'unit', 'unit_number']
    
    for field in floor_fields:
        if field in row.index and pd.notna(row[field]):
            parts.append(f"Floor: {row[field]}")
            break
            
    for field in suite_fields:
        if field in row.index and pd.notna(row[field]):
            parts.append(f"Suite: {row[field]}")
            break
    
    # Size
    size_fields = ['size', 'square_feet', 'sqft', 'area', 'sf']
    for field in size_fields:
        if field in row.index and pd.notna(row[field]):
            parts.append(f"Size: {row[field]} square feet")
            break
    
    # Rent/Price
    rent_fields = ['rent', 'monthly_rent', 'price', 'monthly_price', 'cost']
    for field in rent_fields:
        if field in row.index and pd.notna(row[field]):
            value = str(row[field])
            if not value.startswith('$'):
                value = f"${value}"
            parts.append(f"Monthly Rent: {value}")
            break
    
    # Additional important fields
    important_fields = ['availability', 'status', 'type', 'property_type', 'amenities', 'features']
    for field in important_fields:
        if field in row.index and pd.notna(row[field]):
            parts.append(f"{field.replace('_', ' ').title()}: {row[field]}")
    
    return " | ".join(parts) if parts else "Property information available"


def create_general_structured_description(row: pd.Series, columns: List[str]) -> str:
    """Create a structured description for general CSV data."""
    parts = []
    for col in columns[:10]:  # Limit to first 10 columns to avoid too long text
        if pd.notna(row[col]) and str(row[col]).strip():
            parts.append(f"{col}: {row[col]}")
    return " | ".join(parts)


def create_natural_language_description(row: pd.Series, columns: List[str], is_property_data: bool) -> str:
    """Create a natural language description of the row data."""
    if is_property_data:
        return create_property_natural_description(row)
    else:
        return create_general_natural_description(row, columns)


def create_property_natural_description(row: pd.Series) -> str:
    """Create a natural language description for property data."""
    description_parts = []
    
    # Start with property type or generic
    prop_type = None
    type_fields = ['type', 'property_type', 'space_type']
    for field in type_fields:
        if field in row.index and pd.notna(row[field]):
            prop_type = str(row[field]).lower()
            break
    
    if prop_type:
        description_parts.append(f"This is a {prop_type}")
    else:
        description_parts.append("This property")
    
    # Add location
    address_fields = ['address', 'building_address', 'location']
    for field in address_fields:
        if field in row.index and pd.notna(row[field]):
            description_parts.append(f"located at {row[field]}")
            break
    
    # Add floor and suite info
    floor_info = []
    if 'floor' in row.index and pd.notna(row['floor']):
        floor_info.append(f"on floor {row['floor']}")
    if 'suite' in row.index and pd.notna(row['suite']):
        floor_info.append(f"suite {row['suite']}")
    
    if floor_info:
        description_parts.append(", ".join(floor_info))
    
    # Add size information
    size_fields = ['size', 'square_feet', 'sqft', 'area']
    for field in size_fields:
        if field in row.index and pd.notna(row[field]):
            description_parts.append(f"with {row[field]} square feet of space")
            break
    
    # Add rent information
    rent_fields = ['rent', 'monthly_rent', 'price']
    for field in rent_fields:
        if field in row.index and pd.notna(row[field]):
            rent_value = str(row[field])
            if not rent_value.startswith('$'):
                rent_value = f"${rent_value}"
            description_parts.append(f"available for {rent_value} per month")
            break
    
    # Add amenities or features
    feature_fields = ['amenities', 'features', 'highlights']
    for field in feature_fields:
        if field in row.index and pd.notna(row[field]):
            description_parts.append(f"featuring {row[field]}")
            break
    
    return ". ".join(description_parts) + "."


def create_general_natural_description(row: pd.Series, columns: List[str]) -> str:
    """Create a natural language description for general data."""
    non_null_items = [(col, row[col]) for col in columns if pd.notna(row[col]) and str(row[col]).strip()]
    
    if not non_null_items:
        return "Data record with no significant values."
    
    if len(non_null_items) == 1:
        col, val = non_null_items[0]
        return f"This record has {col} set to {val}."
    
    # Create a natural description
    parts = []
    for i, (col, val) in enumerate(non_null_items[:5]):  # Limit to 5 items
        if i == 0:
            parts.append(f"This record has {col} of {val}")
        elif i == len(non_null_items) - 1:
            parts.append(f"and {col} of {val}")
        else:
            parts.append(f"{col} of {val}")
    
    return ", ".join(parts) + "."


def create_json_representation(row: pd.Series) -> str:
    """Create a clean JSON representation of the row."""
    # Filter out null values and empty strings
    clean_data = {}
    for key, value in row.items():
        if pd.notna(value) and str(value).strip():
            clean_data[key] = value
    
    return json.dumps(clean_data, indent=2, default=str)


def create_key_value_representation(row: pd.Series) -> str:
    """Create a key-value representation optimized for search."""
    kv_pairs = []
    for key, value in row.items():
        if pd.notna(value) and str(value).strip():
            # Create searchable key-value pairs
            clean_key = str(key).replace("_", " ").replace("-", " ").title()
            kv_pairs.append(f"{clean_key}: {value}")
    
    return "\n".join(kv_pairs)


def create_focused_property_documents(row: pd.Series, index: int, file_name: str, user_id: str, base_metadata: dict) -> List[Document]:
    """Create additional focused documents for important property fields."""
    focused_docs = []
    
    # Create focused documents for key searchable fields
    focus_fields = {
        'address': 'Property Address Information',
        'building_address': 'Building Address Information', 
        'location': 'Location Information',
        'rent': 'Rental Price Information',
        'monthly_rent': 'Monthly Rental Information',
        'size': 'Property Size Information',
        'square_feet': 'Square Footage Information',
        'amenities': 'Property Amenities',
        'features': 'Property Features'
    }
    
    for field, description in focus_fields.items():
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            # Create focused text for this field
            focused_text = f"{description}: {row[field]}"
            
            # Add context from other important fields
            context_fields = ['address', 'floor', 'suite', 'size', 'rent']
            context_parts = []
            for ctx_field in context_fields:
                if ctx_field != field and ctx_field in row.index and pd.notna(row[ctx_field]):
                    context_parts.append(f"{ctx_field}: {row[ctx_field]}")
            
            if context_parts:
                focused_text += f"\nContext: {' | '.join(context_parts)}"
            
            # Create metadata for this focused document
            focused_metadata = base_metadata.copy()
            focused_metadata['focus_field'] = field
            focused_metadata['focus_type'] = description
            
            focused_doc = Document(
                text=focused_text,
                doc_id=f"{file_name}_row_{index}_{field}_{user_id}",
                metadata=focused_metadata
            )
            focused_docs.append(focused_doc)
    
    return focused_docs

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
    """Get user-specific VectorStoreIndex."""
    if not user_id:
        return None
    
    # Check cached index first
    if user_id in user_indexes:
        return user_indexes[user_id]
    
    # No cached index found
    return None


async def user_index_exists(user_id: str) -> bool:
    """Return True if the user index exists in cache."""
    return user_id in user_indexes


async def build_user_index(user_id: str, file_paths: List[str]) -> Optional[VectorStoreIndex]:
    """Build user-specific index from file paths with Chroma vector store. Now supports CSV, PDF, TXT, and JSON."""
    try:
        documents = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(path)
                # Create enhanced text embeddings for CSV data
                enhanced_docs = await create_enhanced_csv_embeddings(df, path, user_id)
                documents.extend(enhanced_docs)
                logger.info(f"Created {len(enhanced_docs)} enhanced documents from CSV {os.path.basename(path)}")
            elif ext == '.pdf':
                reader = PdfReader(path)
                text = "\n".join(page.extract_text() or '' for page in reader.pages)
                doc = Document(
                    text=text,
                    doc_id=f"{os.path.basename(path)}_{user_id}",
                    metadata={"user_id": user_id, "file_name": os.path.basename(path), "file_type": "pdf"}
                )
                documents.append(doc)
            elif ext == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc = Document(
                    text=text,
                    doc_id=f"{os.path.basename(path)}_{user_id}",
                    metadata={"user_id": user_id, "file_name": os.path.basename(path), "file_type": "txt"}
                )
                documents.append(doc)
            elif ext == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Flatten JSON to string for now
                text = json.dumps(data, indent=2)
                doc = Document(
                    text=text,
                    doc_id=f"{os.path.basename(path)}_{user_id}",
                    metadata={"user_id": user_id, "file_name": os.path.basename(path), "file_type": "json"}
                )
                documents.append(doc)
            else:
                logger.warning(f"Unsupported file type for {path}")
        
        if documents:
            # Create Chroma vector store for better performance and persistence
            try:
                # Initialize Chroma client
                chroma_client = chromadb.PersistentClient(path=f"user_chroma_db/{user_id}")
                
                # Create or get collection for this user with sanitized name
                sanitized_user_id = sanitize_collection_name(user_id)
                collection_name = f"user_{sanitized_user_id}_collection"
                
                logger.info(f"Creating Chroma collection '{collection_name}' for user '{user_id}'")
                
                try:
                    # Try to delete existing collection to avoid conflicts
                    chroma_client.delete_collection(name=collection_name)
                    logger.info(f"Deleted existing collection '{collection_name}'")
                except:
                    pass  # Collection might not exist
                
                chroma_collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"user_id": user_id, "created_at": str(pd.Timestamp.now())}
                )
                
                # Create ChromaVectorStore
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                
                # Create storage context with Chroma
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Build index with Chroma vector store
                index = VectorStoreIndex.from_documents(
                    documents, 
                    storage_context=storage_context,
                    show_progress=True
                )
                
                logger.info(f"Built Chroma-backed index for user {user_id} with {len(documents)} documents")
                
            except Exception as chroma_error:
                logger.warning(f"Failed to create Chroma vector store for user {user_id}: {chroma_error}")
                logger.info(f"Falling back to in-memory vector store for user {user_id}")
                
                # Fallback to in-memory vector store
                index = VectorStoreIndex.from_documents(documents)
                logger.info(f"Built in-memory index for user {user_id} with {len(documents)} documents")
            
            # Cache the index
            user_indexes[user_id] = index
            
            # Create BM25 retriever for hybrid search
            try:
                # Get nodes from the index for BM25 retriever
                nodes = []
                if hasattr(index, 'docstore') and hasattr(index.docstore, 'docs'):
                    nodes = list(index.docstore.docs.values())
                elif hasattr(index, '_docstore') and hasattr(index._docstore, 'docs'):
                    nodes = list(index._docstore.docs.values())
                else:
                    # Fallback: create nodes from documents
                    from llama_index.core.schema import TextNode
                    nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
                
                if nodes:
                    user_bm25_retrievers[user_id] = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=1,  # Further reduced from 2 to 1 for maximum BM25 speed
                        verbose=False  # Disable verbose logging for performance
                    )
                    logger.info(f"Created BM25 retriever for user {user_id} with {len(nodes)} nodes")
                else:
                    logger.warning(f"No nodes available for BM25 retriever for user {user_id}")
            except Exception as bm25_error:
                logger.error(f"Failed to create BM25 retriever for user {user_id}: {bm25_error}")
                # Continue without BM25 - vector search will still work
            
            return index
        else:
            logger.warning(f"No documents found for user {user_id}")
            return None
    except Exception as e:
        logger.error(f"Error building user index for {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def clear_user_index(user_id: str) -> bool:
    """Clear user-specific index."""
    try:
        # Remove from local cache
        if user_id in user_indexes:
            del user_indexes[user_id]
        if user_id in user_bm25_retrievers:
            del user_bm25_retrievers[user_id]
        
        logger.info(f"Cleared index cache for user: {user_id}")
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
            vector_retriever = user_index.as_retriever(
                similarity_top_k=1,  # Further reduced from 2 to 1 for maximum vector speed
                response_mode="compact"  # Use compact response mode
            )
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
            similarity_top_k=2,  # Further reduced from 3 to 2 for maximum speed
            num_queries=1,  # generate 0 additional queries for speed
            mode=FUSION_MODES.RECIPROCAL_RANK,  # Fast ranking mode
            use_async=True,
            verbose=False,  # Disable verbose logging for performance
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


