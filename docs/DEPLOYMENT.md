# Deployment Guide

This guide covers deploying the Okada Leasing Agent with ChromaDB in different environments.

## Table of Contents

1. [Development Deployment](#development-deployment)
2. [Staging Deployment](#staging-deployment)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Development Deployment

### Local Development Setup

1. **Prerequisites**
   ```bash
   # Install Python 3.8+
   python --version
   
   # Install MongoDB locally or use MongoDB Atlas
   # Install Node.js 16+ (for frontend development)
   ```

2. **Environment Configuration**
   ```bash
   cp env.example .env
   # Edit .env with development values
   ```

3. **Development Environment Variables**
   ```bash
   # .env for development
   GOOGLE_API_KEY=your_dev_api_key
   MONGODB_URI=mongodb://localhost:27017
   MONGO_DATABASE_NAME=okada_leasing_dev
   GOOGLE_CALENDAR_CREDENTIALS_PATH=./credentials.json
   
   # ChromaDB - Local persistent storage
   CHROMA_PERSIST_DIRECTORY=./dev_chroma_db
   CHROMA_HOST=
   CHROMA_PORT=
   CHROMA_COLLECTION_PREFIX=dev_okada_user_
   ```

4. **Run Development Server**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run with hot reload
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Development Testing**
   ```bash
   # Run tests
   pytest
   
   # Check ChromaDB health
   curl http://localhost:8000/api/health/chromadb
   ```

## Staging Deployment

### Staging Environment Setup

1. **Server Requirements**
   - Ubuntu 20.04+ or CentOS 8+
   - 4GB+ RAM
   - 20GB+ disk space
   - Python 3.8+

2. **ChromaDB Server Setup**
   ```bash
   # Install ChromaDB server
   pip install chromadb
   
   # Create ChromaDB data directory
   sudo mkdir -p /opt/chromadb/data
   sudo chown app:app /opt/chromadb/data
   
   # Create systemd service
   sudo tee /etc/systemd/system/chromadb.service << EOF
   [Unit]
   Description=ChromaDB Server
   After=network.target
   
   [Service]
   Type=simple
   User=app
   WorkingDirectory=/opt/chromadb
   ExecStart=/usr/local/bin/chroma run --host 0.0.0.0 --port 8000 --path /opt/chromadb/data
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   EOF
   
   # Start ChromaDB service
   sudo systemctl enable chromadb
   sudo systemctl start chromadb
   ```

3. **Application Deployment**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd okada-leasing-agent
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure environment
   cp env.example .env
   # Edit .env with staging values
   ```

4. **Staging Environment Variables**
   ```bash
   # .env for staging
   GOOGLE_API_KEY=your_staging_api_key
   MONGODB_URI=mongodb://staging-mongo:27017
   MONGO_DATABASE_NAME=okada_leasing_staging
   GOOGLE_CALENDAR_CREDENTIALS_PATH=./credentials.json
   
   # ChromaDB - Remote server
   CHROMA_HOST=localhost
   CHROMA_PORT=8000
   CHROMA_COLLECTION_PREFIX=staging_okada_user_
   ```

5. **Process Management**
   ```bash
   # Install supervisor
   sudo apt install supervisor
   
   # Create supervisor config
   sudo tee /etc/supervisor/conf.d/okada-app.conf << EOF
   [program:okada-app]
   command=/usr/local/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
   directory=/opt/okada-leasing-agent
   user=app
   autostart=true
   autorestart=true
   redirect_stderr=true
   stdout_logfile=/var/log/okada-app.log
   EOF
   
   # Start application
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start okada-app
   ```

## Production Deployment

### Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   App Servers   │    │   ChromaDB      │
│   (nginx/ALB)   │────│   (Multiple)    │────│   Cluster       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                         ┌─────────────────┐
                         │   MongoDB       │
                         │   Replica Set   │
                         └─────────────────┘
```

### High Availability ChromaDB Setup

1. **ChromaDB Cluster Configuration**
   ```bash
   # Server 1 (Primary)
   chroma run --host 0.0.0.0 --port 8000 --path /data/chromadb
   
   # Server 2 (Replica) - Future ChromaDB feature
   # Currently, use multiple instances behind load balancer
   ```

2. **Load Balancer Configuration (nginx)**
   ```nginx
   upstream chromadb_backend {
       server chromadb-1:8000;
       server chromadb-2:8000;
       server chromadb-3:8000;
   }
   
   upstream app_backend {
       server app-1:8000;
       server app-2:8000;
       server app-3:8000;
   }
   
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://app_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Production Environment Variables**
   ```bash
   # .env for production
   GOOGLE_API_KEY=your_production_api_key
   MONGODB_URI=mongodb://prod-mongo-1:27017,prod-mongo-2:27017,prod-mongo-3:27017/okada_leasing?replicaSet=rs0
   MONGO_DATABASE_NAME=okada_leasing_prod
   GOOGLE_CALENDAR_CREDENTIALS_PATH=./credentials.json
   
   # ChromaDB - Production cluster
   CHROMA_HOST=chromadb-cluster.internal
   CHROMA_PORT=8000
   CHROMA_COLLECTION_PREFIX=prod_okada_user_
   ```

### Security Configuration

1. **Firewall Rules**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 80/tcp    # HTTP
   sudo ufw allow 443/tcp   # HTTPS
   sudo ufw deny 8000/tcp   # Block direct access to app
   sudo ufw enable
   ```

2. **SSL/TLS Configuration**
   ```nginx
   server {
       listen 443 ssl http2;
       server_name your-domain.com;
       
       ssl_certificate /etc/ssl/certs/your-domain.crt;
       ssl_certificate_key /etc/ssl/private/your-domain.key;
       
       location / {
           proxy_pass http://app_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

## Docker Deployment

### Docker Compose Setup

1. **Create Docker Compose File**
   ```yaml
   # docker-compose.yml
   version: '3.8'
   
   services:
     chromadb:
       image: chromadb/chroma:latest
       ports:
         - "8000:8000"
       volumes:
         - chroma_data:/chroma/chroma
       environment:
         - CHROMA_SERVER_HOST=0.0.0.0
         - CHROMA_SERVER_PORT=8000
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
         interval: 30s
         timeout: 10s
         retries: 3
   
     mongodb:
       image: mongo:6.0
       ports:
         - "27017:27017"
       volumes:
         - mongo_data:/data/db
       environment:
         - MONGO_INITDB_ROOT_USERNAME=admin
         - MONGO_INITDB_ROOT_PASSWORD=password
   
     app:
       build: .
       ports:
         - "8001:8000"
       depends_on:
         - chromadb
         - mongodb
       environment:
         - CHROMA_HOST=chromadb
         - CHROMA_PORT=8000
         - MONGODB_URI=mongodb://admin:password@mongodb:27017/okada_leasing?authSource=admin
       volumes:
         - ./user_documents:/app/user_documents
         - ./.env:/app/.env
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/api/health/chromadb"]
         interval: 30s
         timeout: 10s
         retries: 3
   
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/ssl
       depends_on:
         - app
   
   volumes:
     chroma_data:
     mongo_data:
   ```

2. **Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000
   
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Deploy with Docker Compose**
   ```bash
   # Build and start services
   docker-compose up -d
   
   # Check service health
   docker-compose ps
   
   # View logs
   docker-compose logs -f app
   
   # Scale application
   docker-compose up -d --scale app=3
   ```

## Cloud Deployment

### AWS Deployment

1. **ECS with Fargate**
   ```yaml
   # ecs-task-definition.json
   {
     "family": "okada-leasing-agent",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "chromadb",
         "image": "chromadb/chroma:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "CHROMA_SERVER_HOST",
             "value": "0.0.0.0"
           }
         ],
         "mountPoints": [
           {
             "sourceVolume": "chroma-data",
             "containerPath": "/chroma/chroma"
           }
         ]
       },
       {
         "name": "app",
         "image": "your-account.dkr.ecr.region.amazonaws.com/okada-leasing-agent:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "CHROMA_HOST",
             "value": "localhost"
           },
           {
             "name": "CHROMA_PORT",
             "value": "8000"
           }
         ],
         "dependsOn": [
           {
             "containerName": "chromadb",
             "condition": "HEALTHY"
           }
         ]
       }
     ],
     "volumes": [
       {
         "name": "chroma-data",
         "efsVolumeConfiguration": {
           "fileSystemId": "fs-12345678"
         }
       }
     ]
   }
   ```

2. **Terraform Configuration**
   ```hcl
   # main.tf
   resource "aws_ecs_cluster" "okada_cluster" {
     name = "okada-leasing-agent"
   }
   
   resource "aws_ecs_service" "okada_service" {
     name            = "okada-leasing-agent"
     cluster         = aws_ecs_cluster.okada_cluster.id
     task_definition = aws_ecs_task_definition.okada_task.arn
     desired_count   = 2
     launch_type     = "FARGATE"
     
     network_configuration {
       subnets         = var.subnet_ids
       security_groups = [aws_security_group.okada_sg.id]
     }
     
     load_balancer {
       target_group_arn = aws_lb_target_group.okada_tg.arn
       container_name   = "app"
       container_port   = 8000
     }
   }
   ```

### Google Cloud Platform

1. **Cloud Run Deployment**
   ```yaml
   # cloudbuild.yaml
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       args: ['build', '-t', 'gcr.io/$PROJECT_ID/okada-leasing-agent', '.']
     - name: 'gcr.io/cloud-builders/docker'
       args: ['push', 'gcr.io/$PROJECT_ID/okada-leasing-agent']
     - name: 'gcr.io/cloud-builders/gcloud'
       args:
         - 'run'
         - 'deploy'
         - 'okada-leasing-agent'
         - '--image'
         - 'gcr.io/$PROJECT_ID/okada-leasing-agent'
         - '--region'
         - 'us-central1'
         - '--platform'
         - 'managed'
         - '--set-env-vars'
         - 'CHROMA_HOST=chromadb-service,CHROMA_PORT=8000'
   ```

## Monitoring and Maintenance

### Health Checks

1. **Application Health**
   ```bash
   # Check application health
   curl http://localhost:8000/docs
   
   # Check ChromaDB health
   curl http://localhost:8000/api/health/chromadb
   
   # Check document indexing status
   curl http://localhost:8000/api/documents/status
   ```

2. **Automated Monitoring**
   ```python
   # monitoring.py
   import requests
   import time
   import logging
   
   def check_health():
       try:
           response = requests.get("http://localhost:8000/api/health/chromadb")
           if response.status_code == 200:
               data = response.json()
               if data["status"] == "healthy":
                   logging.info("ChromaDB is healthy")
               else:
                   logging.error(f"ChromaDB unhealthy: {data['message']}")
           else:
               logging.error(f"Health check failed: {response.status_code}")
       except Exception as e:
           logging.error(f"Health check error: {e}")
   
   # Run every 5 minutes
   while True:
       check_health()
       time.sleep(300)
   ```

### Backup Procedures

1. **ChromaDB Backup**
   ```bash
   # Backup ChromaDB data
   tar -czf chromadb_backup_$(date +%Y%m%d_%H%M%S).tar.gz /path/to/chroma/data
   
   # Automated backup script
   #!/bin/bash
   BACKUP_DIR="/backups/chromadb"
   DATE=$(date +%Y%m%d_%H%M%S)
   
   mkdir -p $BACKUP_DIR
   tar -czf $BACKUP_DIR/chromadb_$DATE.tar.gz /opt/chromadb/data
   
   # Keep only last 7 days of backups
   find $BACKUP_DIR -name "chromadb_*.tar.gz" -mtime +7 -delete
   ```

2. **Data Migration Backup**
   ```python
   # Run before major updates
   from app.migration import DataMigrationManager
   import asyncio
   
   async def create_backup():
       manager = DataMigrationManager()
       user_docs = await manager.discover_existing_documents()
       await manager.backup_existing_collections(list(user_docs.keys()))
   
   asyncio.run(create_backup())
   ```

### Performance Optimization

1. **ChromaDB Tuning**
   ```bash
   # Increase ChromaDB memory
   export CHROMA_SERVER_GRPC_MAX_MESSAGE_SIZE=104857600
   
   # Optimize for read-heavy workloads
   export CHROMA_SERVER_CORS_ALLOW_ORIGINS="*"
   ```

2. **Application Tuning**
   ```python
   # app/rag.py - Optimize batch sizes
   embed_batch_size = 50  # Reduce if memory constrained
   similarity_top_k = 5   # Adjust based on needs
   ```

### Troubleshooting

1. **Common Issues**
   - ChromaDB connection timeouts
   - Memory issues during indexing
   - Slow search performance
   - Data consistency problems

2. **Log Analysis**
   ```bash
   # Application logs
   tail -f /var/log/okada-app.log
   
   # ChromaDB logs
   tail -f /var/log/chromadb.log
   
   # System resources
   htop
   df -h
   ```

3. **Recovery Procedures**
   ```bash
   # Restart services
   sudo systemctl restart chromadb
   sudo systemctl restart okada-app
   
   # Clear and rebuild indexes
   curl -X POST http://localhost:8000/api/reset
   ```

This deployment guide provides comprehensive instructions for deploying the Okada Leasing Agent with ChromaDB in various environments. Choose the deployment method that best fits your infrastructure and requirements. 