# Troubleshooting Guide

This guide covers common issues and solutions for the Okada Leasing Agent with ChromaDB integration.

## Table of Contents

1. [ChromaDB Connection Issues](#chromadb-connection-issues)
2. [Performance Problems](#performance-problems)
3. [Data Consistency Issues](#data-consistency-issues)
4. [Memory and Resource Issues](#memory-and-resource-issues)
5. [Migration Problems](#migration-problems)
6. [API Endpoint Issues](#api-endpoint-issues)
7. [Deployment Issues](#deployment-issues)
8. [Diagnostic Tools](#diagnostic-tools)

## ChromaDB Connection Issues

### Issue: "Failed to connect to ChromaDB"

**Symptoms:**
- Application fails to start
- Health check endpoint returns unhealthy status
- Error messages about ChromaDB connection

**Diagnosis:**
```bash
# Check ChromaDB health
curl http://localhost:8000/api/health/chromadb

# Check if ChromaDB server is running (remote mode)
curl http://your-chromadb-server:8000/api/v1/heartbeat

# Check local ChromaDB directory (local mode)
ls -la ./user_chroma_db/
```

**Solutions:**

1. **Local Mode Issues:**
   ```bash
   # Check directory permissions
   ls -la ./user_chroma_db/
   
   # Create directory if missing
   mkdir -p ./user_chroma_db
   chmod 755 ./user_chroma_db
   
   # Check disk space
   df -h
   ```

2. **Remote Mode Issues:**
   ```bash
   # Test ChromaDB server connectivity
   telnet your-chromadb-server 8000
   
   # Check firewall rules
   sudo ufw status
   
   # Verify environment variables
   echo $CHROMA_HOST
   echo $CHROMA_PORT
   ```

3. **Docker Mode Issues:**
   ```bash
   # Check Docker containers
   docker ps
   
   # Check container logs
   docker logs chromadb-container
   
   # Restart ChromaDB container
   docker restart chromadb-container
   ```

### Issue: "Collection creation failed"

**Symptoms:**
- Document upload fails
- Error messages about collection creation
- Users unable to index documents

**Solutions:**
- Check that your CSV files are properly formatted
- Ensure the user_documents directory exists and has proper permissions
- Restart the application to clear any cached errors
- Check the application logs for specific error messages

## Performance Problems

### Issue: Slow Document Indexing

**Symptoms:**
- Document upload takes very long
- Indexing status shows "in_progress" for extended periods
- High CPU/memory usage during indexing

**Diagnosis:**
```bash
# Monitor system resources
htop
iostat -x 1

# Check indexing status
curl http://localhost:8000/api/documents/status

# Monitor ChromaDB performance
curl http://your-chromadb-server:8000/api/v1/collections
```

**Solutions:**

1. **Reduce Batch Size:**
   ```python
   # In app/rag.py, reduce embed_batch_size
   index = VectorStoreIndex.from_documents(
       documents, 
       vector_store=vector_store,
       embed_batch_size=25  # Reduced from 100
   )
   ```

2. **Optimize CSV Processing:**
   ```python
   # Process large CSV files in chunks
   def process_large_csv(file_path, chunk_size=1000):
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           # Process chunk
           yield chunk
   ```

3. **Hardware Optimization:**
   ```bash
   # Increase system memory
   # Add swap space if needed
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Issue: Slow Search Performance

**Symptoms:**
- Chat responses take long time
- Search timeouts
- High latency in API responses

**Solutions:**

1. **Optimize Search Parameters:**
   ```python
   # In app/rag.py, adjust similarity_top_k
   vector_retriever = index.as_retriever(similarity_top_k=3)  # Reduced from 5
   ```

2. **Index Optimization:**
   ```python
   # Rebuild indexes periodically
   from app.rag import clear_user_index, build_user_index
   
   async def rebuild_user_index(user_id):
       await clear_user_index(user_id)
       # Re-upload documents to rebuild index
   ```

3. **Caching Strategy:**
   ```python
   # Implement result caching
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def cached_search(query, user_id):
       # Implement caching logic
       pass
   ```

## Data Consistency Issues

### Issue: Search Results Don't Match Uploaded Documents

**Symptoms:**
- Users can't find their uploaded documents
- Search returns irrelevant results
- Missing or incorrect property data

**Diagnosis:**
```python
# Check user's indexed documents
from app.rag import get_user_index
import asyncio

async def check_user_documents(user_id):
    index = await get_user_index(user_id)
    if index:
        docstore = index.docstore
        docs = getattr(docstore, "docs", {})
        print(f"User {user_id} has {len(docs)} documents indexed")
        for doc_id, doc in docs.items():
            print(f"  {doc_id}: {doc.text[:100]}...")
    else:
        print(f"No index found for user {user_id}")

asyncio.run(check_user_documents("user@example.com"))
```

**Solutions:**

1. **Re-index User Documents:**
   ```python
   # Clear and rebuild user index
   from app.rag import clear_user_index, build_user_index
   import os
   
   async def reindex_user(user_id):
       # Clear existing index
       await clear_user_index(user_id)
       
       # Find user's documents
       user_doc_dir = os.path.join("user_documents", user_id)
       if os.path.exists(user_doc_dir):
           csv_files = [
               os.path.join(user_doc_dir, f) 
               for f in os.listdir(user_doc_dir) 
               if f.endswith('.csv')
           ]
           
           # Rebuild index
           await build_user_index(user_id, csv_files)
           print(f"Reindexed {len(csv_files)} files for {user_id}")
   ```

2. **Validate Data Format:**
   ```python
   # Check CSV file format
   import pandas as pd
   
   def validate_csv_format(file_path):
       try:
           df = pd.read_csv(file_path)
           print(f"Columns: {list(df.columns)}")
           print(f"Rows: {len(df)}")
           print(f"Sample data:\n{df.head()}")
           return True
       except Exception as e:
           print(f"CSV validation failed: {e}")
           return False
   ```

### Issue: User Isolation Problems

**Symptoms:**
- Users see other users' documents
- Search results contain wrong user's data
- Privacy concerns

**Solutions:**

1. **Verify Collection Isolation:**
   ```python
   # Check user isolation by verifying different users have different data
   from app.rag import get_user_index
   import asyncio
   
   async def check_user_isolation():
       user1_index = await get_user_index("user1@example.com")
       user2_index = await get_user_index("user2@example.com")
       
       print(f"User1 has index: {user1_index is not None}")
       print(f"User2 has index: {user2_index is not None}")
       
       # Verify they are different objects
       if user1_index and user2_index:
           print(f"Indexes are isolated: {user1_index is not user2_index}")
   
   asyncio.run(check_user_isolation())
   ```

2. **Reset User Data:**
   ```bash
   # Reset specific user's data
   curl -X POST "http://localhost:8000/api/reset?user_id=user@example.com"
   ```

## Memory and Resource Issues

### Issue: High Memory Usage

**Symptoms:**
- Application crashes with out-of-memory errors
- System becomes unresponsive
- Docker containers get killed

**Diagnosis:**
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head -10

# Check Docker container memory
docker stats

# Monitor application memory
python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions:**

1. **Optimize Embedding Batch Size:**
   ```python
   # In app/rag.py
   embed_batch_size = 10  # Reduce for low-memory systems
   ```

2. **Process Documents in Chunks:**
   ```python
   # Process large datasets in smaller chunks
   def process_documents_in_chunks(documents, chunk_size=100):
       for i in range(0, len(documents), chunk_size):
           chunk = documents[i:i + chunk_size]
           # Process chunk
           yield chunk
   ```

3. **Increase System Resources:**
   ```bash
   # Add swap space
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   
   # Make permanent
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

### Issue: Disk Space Problems

**Symptoms:**
- ChromaDB fails to store data
- Application crashes during indexing
- "No space left on device" errors

**Solutions:**
```bash
# Check disk usage
df -h

# Clean up old ChromaDB data
find ./user_chroma_db -name "*.log" -mtime +7 -delete

# Compress old backups
find /backups -name "*.tar.gz" -mtime +30 -exec gzip {} \;

# Monitor disk usage
du -sh ./user_chroma_db/*
```

## Migration Problems

### Issue: Migration Fails

**Symptoms:**
- Migration script throws errors
- Some users' data not migrated
- Inconsistent data after migration

**Diagnosis:**
```python
# Check migration status
from app.migration import DataMigrationManager
import asyncio

async def check_migration_status():
    manager = DataMigrationManager()
    
    # Check existing documents
    user_docs = await manager.discover_existing_documents()
    print(f"Found documents for users: {list(user_docs.keys())}")
    
    # Validate ChromaDB connection
    is_healthy = await manager.validate_chromadb_connection()
    print(f"ChromaDB connection healthy: {is_healthy}")

asyncio.run(check_migration_status())
```

**Solutions:**

1. **Manual Migration:**
   ```python
   # Migrate specific user manually
   from app.migration import DataMigrationManager
   import asyncio
   
   async def migrate_user_manually(user_id):
       manager = DataMigrationManager()
       
       # Get user's files
       user_doc_dir = f"user_documents/{user_id}"
       if os.path.exists(user_doc_dir):
           csv_files = [
               os.path.join(user_doc_dir, f)
               for f in os.listdir(user_doc_dir)
               if f.endswith('.csv')
           ]
           
           # Migrate
           success = await manager.migrate_user_documents(user_id, csv_files)
           print(f"Migration success: {success}")
   
   asyncio.run(migrate_user_manually("user@example.com"))
   ```

2. **Rollback Migration:**
   ```python
   # Rollback specific user
   async def rollback_user_migration(user_id):
       manager = DataMigrationManager()
       success = await manager.rollback_migration(user_id)
       print(f"Rollback success: {success}")
   ```

## API Endpoint Issues

### Issue: 503 Service Unavailable

**Symptoms:**
- API endpoints return 503 errors
- "RAG index is not ready" messages
- Users can't chat or upload documents

**Solutions:**

1. **Check Index Status:**
   ```python
   # Check if user has an index
   from app.rag import get_user_index
   import asyncio
   
   async def check_user_index(user_id):
       index = await get_user_index(user_id)
       if index:
           print(f"User {user_id} has a valid index")
       else:
           print(f"User {user_id} has no index - need to upload documents")
   
   asyncio.run(check_user_index("user@example.com"))
   ```

2. **Force Index Creation:**
   ```bash
   # Upload a document to create index
   curl -X POST http://localhost:8000/api/documents/upload \
     -F "user_id=user@example.com" \
     -F "file=@sample.csv"
   ```

### Issue: Slow API Responses

**Symptoms:**
- API calls timeout
- Long response times
- Poor user experience

**Solutions:**

1. **Optimize Database Queries:**
   ```python
   # Add indexes to MongoDB collections
   from app.database import get_database
   import asyncio
   
   async def create_indexes():
       db = await get_database()
       
       # Create indexes for better performance
       await db.users.create_index("email")
       await db.conversation_history.create_index("user_email")
       await db.conversation_history.create_index("timestamp")
   
   asyncio.run(create_indexes())
   ```

2. **Implement Caching:**
   ```python
   # Cache frequently accessed data
   from functools import lru_cache
   import time
   
   @lru_cache(maxsize=100)
   def cached_user_lookup(email):
       # Implement caching logic
       pass
   ```

## Deployment Issues

### Issue: Docker Container Fails to Start

**Symptoms:**
- Docker containers exit immediately
- Connection refused errors
- Service unavailable

**Diagnosis:**
```bash
# Check container status
docker ps -a

# Check container logs
docker logs container-name

# Check Docker network
docker network ls
docker network inspect bridge
```

**Solutions:**

1. **Fix Docker Compose Configuration:**
   ```yaml
   # Ensure proper service dependencies
   services:
     app:
       depends_on:
         - chromadb
         - mongodb
       restart: unless-stopped
   ```

2. **Check Environment Variables:**
   ```bash
   # Verify environment variables in container
   docker exec container-name env | grep CHROMA
   ```

### Issue: Kubernetes Deployment Problems

**Symptoms:**
- Pods crash or restart frequently
- Service discovery issues
- Persistent volume problems

**Solutions:**

1. **Check Pod Status:**
   ```bash
   kubectl get pods
   kubectl describe pod pod-name
   kubectl logs pod-name
   ```

2. **Fix Persistent Volume Claims:**
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: chromadb-pvc
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ```

## Diagnostic Tools

### Health Check Script

```python
#!/usr/bin/env python3
"""
Comprehensive health check script for Okada Leasing Agent
"""

import asyncio
import aiohttp
import sys
import json
from datetime import datetime

async def check_api_health():
    """Check API endpoint health"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/docs') as response:
                if response.status == 200:
                    print("✓ API endpoint is healthy")
                    return True
                else:
                    print(f"✗ API endpoint returned {response.status}")
                    return False
    except Exception as e:
        print(f"✗ API endpoint check failed: {e}")
        return False

async def check_chromadb_health():
    """Check ChromaDB health"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/health/chromadb') as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'healthy':
                        print(f"✓ ChromaDB is healthy ({data.get('collections_count', 0)} collections)")
                        return True
                    else:
                        print(f"✗ ChromaDB is unhealthy: {data.get('message', 'Unknown error')}")
                        return False
                else:
                    print(f"✗ ChromaDB health check returned {response.status}")
                    return False
    except Exception as e:
        print(f"✗ ChromaDB health check failed: {e}")
        return False

async def check_document_indexing():
    """Check document indexing status"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/documents/status') as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    message = data.get('message', 'No message')
                    print(f"✓ Document indexing status: {status} - {message}")
                    return True
                else:
                    print(f"✗ Document status check returned {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Document status check failed: {e}")
        return False

async def main():
    """Run all health checks"""
    print(f"Health Check Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    checks = [
        check_api_health(),
        check_chromadb_health(),
        check_document_indexing()
    ]
    
    results = await asyncio.gather(*checks, return_exceptions=True)
    
    success_count = sum(1 for result in results if result is True)
    total_checks = len(results)
    
    print("=" * 60)
    print(f"Health Check Summary: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print("✓ All systems are healthy")
        sys.exit(0)
    else:
        print("✗ Some systems are unhealthy")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Performance Monitoring Script

```python
#!/usr/bin/env python3
"""
Performance monitoring script for ChromaDB integration
"""

import asyncio
import aiohttp
import time
import psutil
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    async def measure_search_performance(self, query="test property"):
        """Measure search performance"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "user_id": "test@example.com",
                    "message": query
                }
                
                async with session.post(
                    'http://localhost:8000/api/chat',
                    json=payload
                ) as response:
                    await response.json()
                    
                    search_time = time.time() - start_time
                    self.metrics.append({
                        "type": "search",
                        "duration": search_time,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return search_time
        except Exception as e:
            print(f"Search performance test failed: {e}")
            return None
    
    def measure_system_resources(self):
        """Measure system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.metrics.append({
            "type": "system",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "disk_percent": disk.percent,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "cpu": cpu_percent,
            "memory": memory.percent,
            "disk": disk.percent
        }
    
    async def run_performance_test(self, duration=60):
        """Run performance test for specified duration"""
        print(f"Running performance test for {duration} seconds...")
        
        start_time = time.time()
        test_count = 0
        
        while time.time() - start_time < duration:
            # Measure search performance
            search_time = await self.measure_search_performance()
            if search_time:
                print(f"Search {test_count + 1}: {search_time:.2f}s")
            
            # Measure system resources
            resources = self.measure_system_resources()
            
            test_count += 1
            await asyncio.sleep(5)  # Wait 5 seconds between tests
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate performance report"""
        search_metrics = [m for m in self.metrics if m["type"] == "search"]
        system_metrics = [m for m in self.metrics if m["type"] == "system"]
        
        if search_metrics:
            search_times = [m["duration"] for m in search_metrics]
            avg_search_time = sum(search_times) / len(search_times)
            max_search_time = max(search_times)
            min_search_time = min(search_times)
            
            print("\nSearch Performance Report:")
            print(f"  Average search time: {avg_search_time:.2f}s")
            print(f"  Max search time: {max_search_time:.2f}s")
            print(f"  Min search time: {min_search_time:.2f}s")
            print(f"  Total searches: {len(search_times)}")
        
        if system_metrics:
            cpu_usage = [m["cpu_percent"] for m in system_metrics]
            memory_usage = [m["memory_percent"] for m in system_metrics]
            
            print("\nSystem Resource Report:")
            print(f"  Average CPU usage: {sum(cpu_usage) / len(cpu_usage):.1f}%")
            print(f"  Average memory usage: {sum(memory_usage) / len(memory_usage):.1f}%")
            print(f"  Max CPU usage: {max(cpu_usage):.1f}%")
            print(f"  Max memory usage: {max(memory_usage):.1f}%")
        
        # Save detailed metrics to file
        with open(f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

async def main():
    monitor = PerformanceMonitor()
    await monitor.run_performance_test(duration=60)

if __name__ == "__main__":
    asyncio.run(main())
```

Use these diagnostic tools to identify and resolve issues quickly. Run the health check script regularly to monitor system status, and use the performance monitoring script to identify bottlenecks and optimize system performance. 