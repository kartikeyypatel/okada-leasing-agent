# /app/user_context_validator.py
"""
User Context Validation and Debugging Tools

This module provides comprehensive validation and debugging tools for user context
and document association in the RAG chatbot system.
"""

import os
import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from app.config import settings
from app.chroma_client import chroma_manager
import app.rag as rag_module

logger = logging.getLogger(__name__)

@dataclass
class UserContextValidationResult:
    """Result of user context validation."""
    user_id: str
    is_valid: bool
    issues: List[str]
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class DocumentAssociationResult:
    """Result of document association validation."""
    user_id: str
    documents_found: List[str]
    documents_loaded: List[str]
    collection_name: str
    collection_exists: bool
    index_exists: bool
    retriever_available: bool
    issues: List[str]

@dataclass
class UserDebugInfo:
    """Comprehensive debug information for a user."""
    user_id: str
    validation_result: UserContextValidationResult
    document_result: DocumentAssociationResult
    collection_stats: Dict[str, Any]
    index_stats: Dict[str, Any]
    retriever_stats: Dict[str, Any]


class UserContextValidator:
    """Validates and debugs user context and document associations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user ID for safe file system operations."""
        # Remove or replace problematic characters for file system safety
        sanitized = user_id.replace("@", "_at_").replace(".", "_dot_")
        # Additional safety for other problematic characters
        problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
        for char in problematic_chars:
            sanitized = sanitized.replace(char, "_")
        return sanitized
    
    def _generate_collection_name(self, user_id: str) -> str:
        """Generate ChromaDB collection name for a user."""
        return f"{settings.CHROMA_COLLECTION_PREFIX}{hashlib.md5(user_id.encode()).hexdigest()}"
    
    def _get_user_document_path(self, user_id: str) -> str:
        """Get the document directory path for a user."""
        return os.path.join("user_documents", user_id)
    
    async def validate_user_context(self, user_id: str) -> UserContextValidationResult:
        """
        Comprehensive validation of user context handling.
        
        Validates:
        - User ID format and safety
        - Document directory existence and accessibility
        - Collection name generation
        - Path resolution
        """
        issues = []
        details = {}
        
        try:
            # Validate user ID format
            if not user_id or not isinstance(user_id, str):
                issues.append("Invalid user_id: must be a non-empty string")
            else:
                details["user_id_length"] = len(user_id)
                details["user_id_format"] = "email" if "@" in user_id else "other"
            
            # Check if user ID contains problematic characters
            problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            found_problematic = [char for char in problematic_chars if char in user_id]
            if found_problematic:
                issues.append(f"User ID contains problematic characters: {found_problematic}")
            
            # Validate document path generation
            doc_path = self._get_user_document_path(user_id)
            details["document_path"] = doc_path
            details["document_path_exists"] = os.path.exists(doc_path)
            details["document_path_accessible"] = os.access(doc_path, os.R_OK) if os.path.exists(doc_path) else False
            
            # Validate collection name generation
            collection_name = self._generate_collection_name(user_id)
            details["collection_name"] = collection_name
            details["collection_name_length"] = len(collection_name)
            
            # Check collection name validity (ChromaDB requirements)
            if len(collection_name) > 63:
                issues.append(f"Collection name too long: {len(collection_name)} > 63 characters")
            
            # Validate path resolution
            try:
                abs_doc_path = os.path.abspath(doc_path)
                details["absolute_document_path"] = abs_doc_path
                details["path_resolution_success"] = True
            except Exception as e:
                issues.append(f"Path resolution failed: {e}")
                details["path_resolution_success"] = False
            
            # Check for special user ID cases
            if user_id == "ok@gmail.com":
                details["is_test_user"] = True
                # Additional validation for the specific test user
                expected_csv = os.path.join(doc_path, "HackathonInternalKnowledgeBase.csv")
                details["test_csv_exists"] = os.path.exists(expected_csv)
                if not os.path.exists(expected_csv):
                    issues.append("Test user CSV file not found: HackathonInternalKnowledgeBase.csv")
            
            is_valid = len(issues) == 0
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
            is_valid = False
        
        return UserContextValidationResult(
            user_id=user_id,
            is_valid=is_valid,
            issues=issues,
            details=details,
            timestamp=datetime.now()
        )
    
    async def validate_document_association(self, user_id: str) -> DocumentAssociationResult:
        """
        Validate that user documents are properly associated and loaded.
        
        Checks:
        - Document discovery in user directory
        - ChromaDB collection existence
        - Index creation and availability
        - Retriever functionality
        """
        issues = []
        documents_found = []
        documents_loaded = []
        collection_exists = False
        index_exists = False
        retriever_available = False
        
        try:
            # Check for documents in user directory
            doc_path = self._get_user_document_path(user_id)
            if os.path.exists(doc_path):
                all_files = os.listdir(doc_path)
                documents_found = [f for f in all_files if f.endswith('.csv')]
                self.logger.info(f"Found {len(documents_found)} CSV files for user {user_id}")
            else:
                issues.append(f"User document directory does not exist: {doc_path}")
            
            # Check ChromaDB collection
            collection_name = self._generate_collection_name(user_id)
            try:
                collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
                collection_exists = True
                
                # Check if collection has documents
                try:
                    count = await asyncio.to_thread(collection.count)
                    if count > 0:
                        documents_loaded = [f"ChromaDB documents: {count}"]
                        self.logger.info(f"ChromaDB collection for {user_id} contains {count} documents")
                    else:
                        issues.append("ChromaDB collection exists but is empty")
                except Exception as e:
                    issues.append(f"Failed to count documents in collection: {e}")
                    
            except Exception as e:
                issues.append(f"Failed to access ChromaDB collection: {e}")
                collection_exists = False
            
            # Check if user index exists
            try:
                user_index = await rag_module.get_user_index(user_id)
                index_exists = user_index is not None
                if not index_exists:
                    issues.append("User index not available")
                else:
                    self.logger.info(f"User index exists for {user_id}")
            except Exception as e:
                issues.append(f"Failed to get user index: {e}")
                index_exists = False
            
            # Check if retriever can be created
            try:
                fusion_retriever = rag_module.get_fusion_retriever(user_id)
                retriever_available = fusion_retriever is not None
                if not retriever_available:
                    issues.append("Fusion retriever not available")
                else:
                    self.logger.info(f"Fusion retriever available for {user_id}")
            except Exception as e:
                issues.append(f"Failed to create fusion retriever: {e}")
                retriever_available = False
            
            # Cross-validation: if documents exist but not loaded
            if documents_found and not documents_loaded:
                issues.append("Documents found in directory but not loaded into ChromaDB")
            
            # Cross-validation: if collection exists but no index
            if collection_exists and not index_exists:
                issues.append("ChromaDB collection exists but no user index available")
            
        except Exception as e:
            issues.append(f"Document association validation error: {e}")
        
        return DocumentAssociationResult(
            user_id=user_id,
            documents_found=documents_found,
            documents_loaded=documents_loaded,
            collection_name=collection_name,
            collection_exists=collection_exists,
            index_exists=index_exists,
            retriever_available=retriever_available,
            issues=issues
        )
    
    async def get_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed statistics about user's ChromaDB collection."""
        stats = {}
        
        try:
            collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
            
            # Basic stats
            stats["collection_name"] = collection.name
            stats["document_count"] = await asyncio.to_thread(collection.count)
            
            # Try to get sample documents
            try:
                if stats["document_count"] > 0:
                    sample_results = await asyncio.to_thread(
                        collection.peek, 
                        limit=min(3, stats["document_count"])
                    )
                    stats["sample_documents"] = len(sample_results.get("documents", []))
                    stats["sample_metadata"] = sample_results.get("metadatas", [])[:1]  # First metadata sample
                else:
                    stats["sample_documents"] = 0
                    stats["sample_metadata"] = []
            except Exception as e:
                stats["sample_error"] = str(e)
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    async def get_index_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed statistics about user's index."""
        stats = {}
        
        try:
            user_index = await rag_module.get_user_index(user_id)
            
            if user_index:
                stats["index_available"] = True
                
                # Check docstore
                if hasattr(user_index, 'docstore') and hasattr(user_index.docstore, 'docs'):
                    docs = user_index.docstore.docs
                    stats["docstore_document_count"] = len(docs) if docs else 0
                    
                    # Sample document info
                    if docs:
                        sample_doc = next(iter(docs.values()))
                        stats["sample_doc_id"] = sample_doc.doc_id if hasattr(sample_doc, 'doc_id') else "unknown"
                        stats["sample_doc_metadata_keys"] = list(sample_doc.metadata.keys()) if hasattr(sample_doc, 'metadata') else []
                else:
                    stats["docstore_document_count"] = "unknown"
                
                # Check vector store
                if hasattr(user_index, 'vector_store'):
                    stats["vector_store_type"] = type(user_index.vector_store).__name__
                else:
                    stats["vector_store_type"] = "unknown"
                    
            else:
                stats["index_available"] = False
                stats["error"] = "Index not found"
                
        except Exception as e:
            stats["error"] = str(e)
            stats["index_available"] = False
        
        return stats
    
    async def get_retriever_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed statistics about user's retrievers."""
        stats = {}
        
        try:
            # Check fusion retriever
            fusion_retriever = rag_module.get_fusion_retriever(user_id)
            stats["fusion_retriever_available"] = fusion_retriever is not None
            
            if fusion_retriever:
                stats["fusion_retriever_type"] = type(fusion_retriever).__name__
                
                # Check individual retrievers
                if hasattr(fusion_retriever, '_retrievers'):
                    retrievers = fusion_retriever._retrievers
                    stats["individual_retriever_count"] = len(retrievers)
                    stats["individual_retriever_types"] = [type(r).__name__ for r in retrievers]
                else:
                    stats["individual_retriever_count"] = "unknown"
            
            # Check BM25 retriever specifically
            user_bm25 = rag_module.user_bm25_retrievers.get(user_id)
            stats["bm25_retriever_available"] = user_bm25 is not None
            
            if user_bm25:
                stats["bm25_retriever_type"] = type(user_bm25).__name__
                # Try to get node count if available
                if hasattr(user_bm25, '_nodes'):
                    stats["bm25_node_count"] = len(user_bm25._nodes)
                else:
                    stats["bm25_node_count"] = "unknown"
            
            # Check user index retriever
            user_index = await rag_module.get_user_index(user_id)
            if user_index:
                try:
                    vector_retriever = user_index.as_retriever(similarity_top_k=1)
                    stats["vector_retriever_available"] = True
                    stats["vector_retriever_type"] = type(vector_retriever).__name__
                except Exception as e:
                    stats["vector_retriever_available"] = False
                    stats["vector_retriever_error"] = str(e)
            else:
                stats["vector_retriever_available"] = False
                
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    async def get_comprehensive_debug_info(self, user_id: str) -> UserDebugInfo:
        """Get comprehensive debug information for a user."""
        self.logger.info(f"Generating comprehensive debug info for user: {user_id}")
        
        # Run all validations and stats gathering
        validation_result = await self.validate_user_context(user_id)
        document_result = await self.validate_document_association(user_id)
        collection_stats = await self.get_collection_stats(user_id)
        index_stats = await self.get_index_stats(user_id)
        retriever_stats = await self.get_retriever_stats(user_id)
        
        return UserDebugInfo(
            user_id=user_id,
            validation_result=validation_result,
            document_result=document_result,
            collection_stats=collection_stats,
            index_stats=index_stats,
            retriever_stats=retriever_stats
        )
    
    async def fix_user_context_issues(self, user_id: str) -> Dict[str, Any]:
        """
        Attempt to fix common user context issues.
        
        Returns a report of what was attempted and the results.
        """
        fix_report = {
            "user_id": user_id,
            "fixes_attempted": [],
            "fixes_successful": [],
            "fixes_failed": [],
            "final_status": {}
        }
        
        try:
            # Get initial debug info
            debug_info = await self.get_comprehensive_debug_info(user_id)
            
            # Fix 1: Rebuild index if documents exist but index is missing
            if (debug_info.document_result.documents_found and 
                not debug_info.document_result.index_exists):
                
                fix_report["fixes_attempted"].append("rebuild_index")
                try:
                    doc_path = self._get_user_document_path(user_id)
                    file_paths = [os.path.join(doc_path, f) for f in debug_info.document_result.documents_found]
                    
                    new_index = await rag_module.build_user_index(user_id, file_paths)
                    if new_index:
                        fix_report["fixes_successful"].append("rebuild_index")
                        self.logger.info(f"Successfully rebuilt index for user {user_id}")
                    else:
                        fix_report["fixes_failed"].append("rebuild_index: returned None")
                except Exception as e:
                    fix_report["fixes_failed"].append(f"rebuild_index: {e}")
            
            # Fix 2: Clear and recreate collection if it's corrupted
            if (debug_info.document_result.collection_exists and 
                not debug_info.document_result.documents_loaded):
                
                fix_report["fixes_attempted"].append("recreate_collection")
                try:
                    # Clear existing collection
                    await rag_module.clear_user_index(user_id)
                    
                    # Rebuild if documents exist
                    if debug_info.document_result.documents_found:
                        doc_path = self._get_user_document_path(user_id)
                        file_paths = [os.path.join(doc_path, f) for f in debug_info.document_result.documents_found]
                        
                        new_index = await rag_module.build_user_index(user_id, file_paths)
                        if new_index:
                            fix_report["fixes_successful"].append("recreate_collection")
                            self.logger.info(f"Successfully recreated collection for user {user_id}")
                        else:
                            fix_report["fixes_failed"].append("recreate_collection: rebuild failed")
                    else:
                        fix_report["fixes_successful"].append("recreate_collection")
                        
                except Exception as e:
                    fix_report["fixes_failed"].append(f"recreate_collection: {e}")
            
            # Fix 3: Specific fix for "ok@gmail.com" user - ensure test data is properly loaded
            if user_id == "ok@gmail.com":
                fix_report["fixes_attempted"].append("validate_test_user_data")
                try:
                    await self._validate_and_fix_test_user_data(user_id, fix_report)
                except Exception as e:
                    fix_report["fixes_failed"].append(f"validate_test_user_data: {e}")
            
            # Fix 4: Validate collection name generation consistency
            fix_report["fixes_attempted"].append("validate_collection_naming")
            try:
                expected_collection_name = self._generate_collection_name(user_id)
                collection_stats = debug_info.collection_stats
                
                if "collection_name" in collection_stats:
                    actual_name = collection_stats["collection_name"]
                    if actual_name != expected_collection_name:
                        self.logger.warning(f"Collection name mismatch for {user_id}: expected {expected_collection_name}, got {actual_name}")
                        fix_report["fixes_failed"].append(f"collection_naming: name mismatch - expected {expected_collection_name}, got {actual_name}")
                    else:
                        fix_report["fixes_successful"].append("validate_collection_naming")
                else:
                    fix_report["fixes_failed"].append("validate_collection_naming: no collection name in stats")
                    
            except Exception as e:
                fix_report["fixes_failed"].append(f"validate_collection_naming: {e}")
            
            # Get final status
            final_debug_info = await self.get_comprehensive_debug_info(user_id)
            fix_report["final_status"] = {
                "context_valid": final_debug_info.validation_result.is_valid,
                "documents_loaded": len(final_debug_info.document_result.documents_loaded) > 0,
                "index_exists": final_debug_info.document_result.index_exists,
                "retriever_available": final_debug_info.document_result.retriever_available,
                "remaining_issues": (final_debug_info.validation_result.issues + 
                                   final_debug_info.document_result.issues)
            }
            
        except Exception as e:
            fix_report["fixes_failed"].append(f"fix_process_error: {e}")
        
        return fix_report
    
    async def _validate_and_fix_test_user_data(self, user_id: str, fix_report: Dict[str, Any]):
        """
        Specific validation and fixes for the test user "ok@gmail.com".
        
        This function ensures that:
        1. The test CSV file exists and is readable
        2. The "84 Mulberry St" test data is present
        3. The collection and index properly contain this data
        """
        try:
            # Check if test CSV exists
            doc_path = self._get_user_document_path(user_id)
            test_csv_path = os.path.join(doc_path, "HackathonInternalKnowledgeBase.csv")
            
            if not os.path.exists(test_csv_path):
                # Try to find the CSV in sample_docs directory
                sample_csv_path = os.path.join("sample_docs", "HackathonInternalKnowledgeBase.csv")
                if os.path.exists(sample_csv_path):
                    # Copy the sample CSV to user directory
                    import shutil
                    os.makedirs(doc_path, exist_ok=True)
                    shutil.copy2(sample_csv_path, test_csv_path)
                    self.logger.info(f"Copied sample CSV to user directory for {user_id}")
                    fix_report["fixes_successful"].append("copy_test_csv")
                else:
                    fix_report["fixes_failed"].append("test_csv_not_found: neither in user dir nor sample_docs")
                    return
            
            # Validate CSV content
            try:
                df = pd.read_csv(test_csv_path)
                
                # Check for "84 Mulberry St" in the data
                test_address_found = False
                for col in df.columns:
                    if df[col].astype(str).str.contains("84 Mulberry", case=False, na=False).any():
                        test_address_found = True
                        matching_rows = df[df[col].astype(str).str.contains("84 Mulberry", case=False, na=False)]
                        self.logger.info(f"Found test address '84 Mulberry St' in column '{col}' for user {user_id}")
                        self.logger.debug(f"Matching rows: {len(matching_rows)}")
                        break
                
                if test_address_found:
                    fix_report["fixes_successful"].append("validate_test_data_present")
                else:
                    fix_report["fixes_failed"].append("validate_test_data: 84 Mulberry St not found in CSV")
                    return
                    
            except Exception as csv_error:
                fix_report["fixes_failed"].append(f"validate_csv_content: {csv_error}")
                return
            
            # Ensure the index is built with this data
            try:
                current_index = await rag_module.get_user_index(user_id)
                if not current_index:
                    # Build the index
                    new_index = await rag_module.build_user_index(user_id, [test_csv_path])
                    if new_index:
                        fix_report["fixes_successful"].append("build_test_user_index")
                        self.logger.info(f"Successfully built index for test user {user_id}")
                    else:
                        fix_report["fixes_failed"].append("build_test_user_index: returned None")
                        return
                else:
                    fix_report["fixes_successful"].append("test_user_index_exists")
                
                # Test retriever functionality with the test query
                fusion_retriever = rag_module.get_fusion_retriever(user_id)
                if fusion_retriever:
                    try:
                        test_results = await fusion_retriever.aretrieve("84 Mulberry St")
                        if test_results and len(test_results) > 0:
                            fix_report["fixes_successful"].append("test_query_retrieval")
                            self.logger.info(f"Test query '84 Mulberry St' returned {len(test_results)} results for user {user_id}")
                        else:
                            fix_report["fixes_failed"].append("test_query_retrieval: no results for 84 Mulberry St")
                    except Exception as retrieval_error:
                        fix_report["fixes_failed"].append(f"test_query_retrieval: {retrieval_error}")
                else:
                    fix_report["fixes_failed"].append("test_retriever_creation: fusion retriever not available")
                    
            except Exception as index_error:
                fix_report["fixes_failed"].append(f"test_user_index_handling: {index_error}")
                
        except Exception as e:
            fix_report["fixes_failed"].append(f"test_user_validation: {e}")
    
    async def validate_specific_user_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Validate that a specific query works correctly for a user.
        
        This is useful for testing specific cases like "84 Mulberry St" for "ok@gmail.com".
        """
        validation_result = {
            "user_id": user_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "validation_steps": {},
            "overall_success": False
        }
        
        try:
            # Step 1: Validate user context
            context_validation = await self.validate_user_context(user_id)
            validation_result["validation_steps"]["user_context"] = {
                "success": context_validation.is_valid,
                "issues": context_validation.issues
            }
            
            # Step 2: Check if user has index
            user_index = await rag_module.get_user_index(user_id)
            validation_result["validation_steps"]["user_index"] = {
                "success": user_index is not None,
                "details": "Index available" if user_index else "Index not available"
            }
            
            if not user_index:
                validation_result["overall_success"] = False
                return validation_result
            
            # Step 3: Test fusion retriever creation
            fusion_retriever = rag_module.get_fusion_retriever(user_id)
            validation_result["validation_steps"]["fusion_retriever"] = {
                "success": fusion_retriever is not None,
                "details": "Retriever available" if fusion_retriever else "Retriever not available"
            }
            
            if not fusion_retriever:
                validation_result["overall_success"] = False
                return validation_result
            
            # Step 4: Test the specific query
            try:
                search_results = await fusion_retriever.aretrieve(query)
                validation_result["validation_steps"]["query_execution"] = {
                    "success": True,
                    "results_count": len(search_results),
                    "results_preview": [
                        {
                            "score": float(node.score) if hasattr(node, 'score') else None,
                            "content_preview": node.get_content()[:200] + "..." if len(node.get_content()) > 200 else node.get_content()
                        }
                        for node in search_results[:3]  # First 3 results
                    ]
                }
                
                # Check if results are relevant (for address queries, look for the address in results)
                if "mulberry" in query.lower():
                    relevant_results = [
                        node for node in search_results 
                        if "mulberry" in node.get_content().lower()
                    ]
                    validation_result["validation_steps"]["relevance_check"] = {
                        "success": len(relevant_results) > 0,
                        "relevant_results": len(relevant_results),
                        "total_results": len(search_results)
                    }
                else:
                    validation_result["validation_steps"]["relevance_check"] = {
                        "success": len(search_results) > 0,
                        "note": "Generic relevance check - results found"
                    }
                
            except Exception as query_error:
                validation_result["validation_steps"]["query_execution"] = {
                    "success": False,
                    "error": str(query_error)
                }
            
            # Overall success assessment
            all_steps_successful = all(
                step.get("success", False) 
                for step in validation_result["validation_steps"].values()
            )
            validation_result["overall_success"] = all_steps_successful
            
        except Exception as e:
            validation_result["validation_steps"]["validation_error"] = {
                "success": False,
                "error": str(e)
            }
            validation_result["overall_success"] = False
        
        return validation_result


# Global instance for easy access
user_context_validator = UserContextValidator()