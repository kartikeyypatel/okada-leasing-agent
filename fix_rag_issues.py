#!/usr/bin/env python3
"""
Direct fix for RAG issues - rebuild everything properly
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def fix_rag_issues():
    """Fix all RAG issues by rebuilding everything properly"""
    try:
        from app.rag import clear_user_index, user_indexes, user_bm25_retrievers
        from app.chroma_client import chroma_manager
        import pandas as pd
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.schema import TextNode
        
        user_id = "ok@gmail.com"
        user_docs_path = f"user_documents/{user_id}"
        
        print(f"🔧 Fixing RAG issues for user: {user_id}")
        
        # Step 1: Clear everything
        print("🧹 Clearing existing data...")
        await clear_user_index(user_id)
        
        # Step 2: Load CSV data
        csv_files = [f for f in os.listdir(user_docs_path) if f.endswith('.csv')]
        if not csv_files:
            print("❌ No CSV files found")
            return False
        
        file_paths = [os.path.join(user_docs_path, f) for f in csv_files]
        print(f"📄 Processing {len(csv_files)} CSV files")
        
        # Step 3: Create documents
        documents = []
        for path in file_paths:
            df = pd.read_csv(path)
            df.columns = [col.strip().lower() for col in df.columns]
            
            for index, row in df.iterrows():
                row_content = ", ".join([f"{header}: {value}" for header, value in row.items() if pd.notna(value)])
                
                metadata = {
                    "user_id": user_id,
                    "file_name": os.path.basename(path),
                    "row_index": index,
                    "upload_timestamp": pd.Timestamp.now().isoformat()
                }
                
                for key, value in row.to_dict().items():
                    if pd.notna(value):
                        metadata[key] = str(value)
                
                doc = Document(
                    text=row_content, 
                    doc_id=f"{os.path.basename(path)}_row_{index}_{user_id}",
                    metadata=metadata
                )
                documents.append(doc)
        
        print(f"📊 Created {len(documents)} documents")
        
        # Step 4: Create ChromaDB index
        try:
            collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            
            index = VectorStoreIndex.from_documents(
                documents, 
                vector_store=vector_store,
                embed_batch_size=50  # Smaller batch size
            )
            
            user_indexes[user_id] = index
            print("✅ ChromaDB index created successfully")
            
        except Exception as e:
            print(f"⚠️  ChromaDB failed, using in-memory: {e}")
            index = VectorStoreIndex.from_documents(documents, embed_batch_size=50)
            user_indexes[user_id] = index
            print("✅ In-memory index created successfully")
        
        # Step 5: Create BM25 retriever from documents
        try:
            # Convert documents to TextNodes for BM25
            nodes = []
            for doc in documents:
                node = TextNode(
                    text=doc.text,
                    metadata=doc.metadata,
                    id_=doc.doc_id
                )
                nodes.append(node)
            
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=5
            )
            
            user_bm25_retrievers[user_id] = bm25_retriever
            print(f"✅ BM25 retriever created with {len(nodes)} nodes")
            
        except Exception as e:
            print(f"❌ Failed to create BM25 retriever: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 6: Test everything
        print("\n🧪 Testing the fixed system...")
        
        # Test fusion retriever
        from app.rag import get_fusion_retriever
        fusion_retriever = get_fusion_retriever(user_id)
        
        if fusion_retriever:
            print("✅ Fusion retriever created successfully")
            
            # Test search
            test_results = await fusion_retriever.aretrieve("36 W 36th St")
            print(f"✅ Test search returned {len(test_results)} results")
            
            if len(test_results) > 0:
                sample_result = test_results[0]
                print(f"📝 Sample result: {sample_result.text[:150]}...")
            
        else:
            print("❌ Failed to create fusion retriever")
            return False
        
        # Test response generation
        from app.strict_response_generator import StrictResponseGenerator
        generator = StrictResponseGenerator()
        
        test_query = "tell me about 36 W 36th St"
        strict_result = await generator.generate_strict_response(
            user_query=test_query,
            retrieved_nodes=test_results,
            user_id=user_id
        )
        
        print(f"✅ Response generation successful: {strict_result.generation_successful}")
        print(f"📝 Sample response: {strict_result.response_text[:200]}...")
        
        print("\n🎉 RAG system fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing RAG issues: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(fix_rag_issues())
    sys.exit(0 if success else 1)