# /app/security_compliance_manager.py
"""
Security and Compliance Management System

This module provides comprehensive security monitoring and compliance validation
for MongoDB and ChromaDB access, ensuring data protection and regulatory compliance.
"""

import asyncio
import logging
import hashlib
import hmac
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

from app.database import db_manager
from app.models import SecurityStatus, EncryptedData, AccessAuditReport, AnomalyReport, ComplianceReport

logger = logging.getLogger(__name__)


class AccessType(Enum):
    """Types of database access operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    QUERY = "query"
    INDEX = "index"


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"


@dataclass
class AccessLogEntry:
    """Individual access log entry."""
    timestamp: datetime
    user_id: str
    access_type: AccessType
    collection_name: str
    operation: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    data_accessed: Optional[Dict[str, Any]] = None
    sensitive_fields: List[str] = field(default_factory=list)


@dataclass
class SecurityAlert:
    """Security alert for anomalous activity."""
    timestamp: datetime
    alert_type: str
    severity: str
    user_id: str
    description: str
    evidence: Dict[str, Any]
    recommended_action: str
    auto_resolved: bool = False


class SecurityComplianceManager:
    """
    Comprehensive security and compliance monitoring system.
    
    Provides connection security validation, data encryption, access auditing,
    anomaly detection, and compliance monitoring for GDPR/CCPA requirements.
    """
    
    def __init__(self):
        self.access_logs: List[AccessLogEntry] = []
        self.security_alerts: List[SecurityAlert] = []
        self.encryption_key = self._initialize_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Security monitoring settings
        self.anomaly_thresholds = {
            "max_requests_per_minute": 100,
            "max_failed_attempts": 5,
            "unusual_access_hours": (22, 6),  # 10 PM to 6 AM
            "max_data_volume_mb": 100,
            "suspicious_query_patterns": [
                r".*\$ne.*null.*",  # Potential injection
                r".*\$regex.*\.\*.*",  # Overly broad regex
                r".*\$where.*",  # JavaScript execution
            ]
        }
        
        # Compliance settings
        self.sensitive_field_patterns = [
            r".*email.*",
            r".*phone.*",
            r".*ssn.*",
            r".*credit.*card.*",
            r".*password.*",
            r".*token.*",
            r".*key.*"
        ]
        
        # Access pattern tracking
        self.user_access_patterns: Dict[str, List[AccessLogEntry]] = {}
        self.ip_access_patterns: Dict[str, List[AccessLogEntry]] = {}
        
        logger.info("Security and Compliance Manager initialized")
    
    async def validate_connection_security(self) -> SecurityStatus:
        """
        Validate MongoDB and ChromaDB connection security.
        
        Returns:
            SecurityStatus with connection security assessment
        """
        try:
            security_issues = []
            
            # Check MongoDB connection security
            if db_manager.client:
                # Verify TLS/SSL connection
                server_info = await db_manager.client.server_info()
                
                # Check if connection is encrypted
                connection_secure = True
                if not self._is_connection_encrypted():
                    security_issues.append("MongoDB connection is not encrypted")
                    connection_secure = False
                
                # Check authentication
                if not self._is_authentication_enabled():
                    security_issues.append("MongoDB authentication is not properly configured")
                    connection_secure = False
                
                # Check database permissions
                permissions_valid = await self._validate_database_permissions()
                if not permissions_valid:
                    security_issues.append("Database permissions are overly permissive")
                    connection_secure = False
            else:
                security_issues.append("MongoDB connection not available for security validation")
                connection_secure = False
            
            # Check ChromaDB security
            chroma_secure = self._validate_chromadb_security()
            if not chroma_secure:
                security_issues.append("ChromaDB security configuration needs review")
            
            # Determine overall security status
            if not security_issues:
                status = SecurityStatus.SECURE
            elif len(security_issues) <= 2:
                status = SecurityStatus.WARNING
            else:
                status = SecurityStatus.VULNERABLE
            
            logger.info(f"Connection security validation completed: {status.value}")
            
            return SecurityStatus(status.value)
            
        except Exception as e:
            logger.error(f"Error validating connection security: {e}")
            return SecurityStatus.UNKNOWN
    
    async def encrypt_sensitive_data(self, data: Dict[str, Any]) -> EncryptedData:
        """
        Encrypt sensitive data fields before storage.
        
        Args:
            data: Dictionary containing potentially sensitive data
            
        Returns:
            EncryptedData with encrypted sensitive fields
        """
        try:
            encrypted_fields = {}
            sensitive_fields = []
            
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    if isinstance(value, str):
                        # Encrypt the sensitive field
                        encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
                        encrypted_fields[key] = encrypted_value
                        sensitive_fields.append(key)
                    else:
                        # Convert to string and encrypt
                        encrypted_value = self.cipher_suite.encrypt(str(value).encode()).decode()
                        encrypted_fields[key] = encrypted_value
                        sensitive_fields.append(key)
                else:
                    # Keep non-sensitive fields as-is
                    encrypted_fields[key] = value
            
            return EncryptedData(
                encrypted_data=encrypted_fields,
                sensitive_fields=sensitive_fields,
                encryption_timestamp=datetime.now(),
                encryption_algorithm="Fernet"
            )
            
        except Exception as e:
            logger.error(f"Error encrypting sensitive data: {e}")
            raise
    
    async def decrypt_sensitive_data(self, encrypted_data: EncryptedData) -> Dict[str, Any]:
        """
        Decrypt sensitive data fields.
        
        Args:
            encrypted_data: EncryptedData object with encrypted fields
            
        Returns:
            Dictionary with decrypted sensitive fields
        """
        try:
            decrypted_data = encrypted_data.encrypted_data.copy()
            
            for field in encrypted_data.sensitive_fields:
                if field in decrypted_data:
                    encrypted_value = decrypted_data[field]
                    if isinstance(encrypted_value, str):
                        # Decrypt the field
                        decrypted_value = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
                        decrypted_data[field] = decrypted_value
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting sensitive data: {e}")
            raise
    
    async def audit_access_patterns(self) -> AccessAuditReport:
        """
        Audit database access patterns and generate comprehensive report.
        
        Returns:
            AccessAuditReport with access statistics and patterns
        """
        try:
            # Analyze access patterns from the last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_accesses = [
                log for log in self.access_logs
                if log.timestamp >= cutoff_time
            ]
            
            # Calculate access statistics
            total_accesses = len(recent_accesses)
            successful_accesses = len([log for log in recent_accesses if log.success])
            failed_accesses = total_accesses - successful_accesses
            
            # Analyze by access type
            access_by_type = {}
            for access_type in AccessType:
                count = len([log for log in recent_accesses if log.access_type == access_type])
                access_by_type[access_type.value] = count
            
            # Analyze by user
            user_access_counts = {}
            for log in recent_accesses:
                user_access_counts[log.user_id] = user_access_counts.get(log.user_id, 0) + 1
            
            # Analyze by collection
            collection_access_counts = {}
            for log in recent_accesses:
                collection_access_counts[log.collection_name] = collection_access_counts.get(log.collection_name, 0) + 1
            
            # Identify top users and collections
            top_users = sorted(user_access_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_collections = sorted(collection_access_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate success rate
            success_rate = (successful_accesses / total_accesses) if total_accesses > 0 else 1.0
            
            return AccessAuditReport(
                report_period="24_hours",
                total_accesses=total_accesses,
                successful_accesses=successful_accesses,
                failed_accesses=failed_accesses,
                success_rate=success_rate,
                access_by_type=access_by_type,
                top_users=dict(top_users),
                top_collections=dict(top_collections),
                unique_users=len(user_access_counts),
                unique_ips=len(self.ip_access_patterns),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error auditing access patterns: {e}")
            return AccessAuditReport(
                report_period="error",
                total_accesses=0,
                successful_accesses=0,
                failed_accesses=0,
                success_rate=0.0,
                access_by_type={},
                top_users={},
                top_collections={},
                unique_users=0,
                unique_ips=0,
                timestamp=datetime.now()
            )
    
    async def detect_anomalous_access(self) -> AnomalyReport:
        """
        Detect anomalous access patterns and security threats.
        
        Returns:
            AnomalyReport with detected anomalies and recommendations
        """
        try:
            anomalies = []
            
            # Check for unusual access patterns in the last hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_accesses = [
                log for log in self.access_logs
                if log.timestamp >= cutoff_time
            ]
            
            # 1. Check for excessive requests from single user
            user_request_counts = {}
            for log in recent_accesses:
                user_request_counts[log.user_id] = user_request_counts.get(log.user_id, 0) + 1
            
            for user_id, count in user_request_counts.items():
                if count > self.anomaly_thresholds["max_requests_per_minute"] * 60:
                    anomalies.append({
                        "type": "excessive_requests",
                        "user_id": user_id,
                        "description": f"User {user_id} made {count} requests in the last hour",
                        "severity": "high",
                        "recommended_action": "Review user activity and consider rate limiting"
                    })
            
            # 2. Check for failed authentication attempts
            failed_attempts = {}
            for log in recent_accesses:
                if not log.success and log.error_message:
                    failed_attempts[log.user_id] = failed_attempts.get(log.user_id, 0) + 1
            
            for user_id, count in failed_attempts.items():
                if count > self.anomaly_thresholds["max_failed_attempts"]:
                    anomalies.append({
                        "type": "excessive_failed_attempts",
                        "user_id": user_id,
                        "description": f"User {user_id} had {count} failed attempts in the last hour",
                        "severity": "critical",
                        "recommended_action": "Investigate potential brute force attack and consider blocking user"
                    })
            
            # 3. Check for unusual access hours
            current_hour = datetime.now().hour
            unusual_start, unusual_end = self.anomaly_thresholds["unusual_access_hours"]
            
            if unusual_start <= current_hour or current_hour <= unusual_end:
                recent_unusual_access = [
                    log for log in recent_accesses
                    if unusual_start <= log.timestamp.hour or log.timestamp.hour <= unusual_end
                ]
                
                if len(recent_unusual_access) > 10:  # Threshold for unusual hour activity
                    anomalies.append({
                        "type": "unusual_access_hours",
                        "description": f"{len(recent_unusual_access)} accesses during unusual hours ({unusual_start}-{unusual_end})",
                        "severity": "medium",
                        "recommended_action": "Review after-hours access patterns and verify legitimacy"
                    })
            
            # 4. Check for suspicious query patterns
            for log in recent_accesses:
                if log.operation:
                    for pattern in self.anomaly_thresholds["suspicious_query_patterns"]:
                        if re.search(pattern, log.operation, re.IGNORECASE):
                            anomalies.append({
                                "type": "suspicious_query_pattern",
                                "user_id": log.user_id,
                                "description": f"Suspicious query pattern detected: {pattern}",
                                "severity": "high",
                                "recommended_action": "Review query for potential injection attack"
                            })
                            break
            
            # Generate security alerts for critical anomalies
            for anomaly in anomalies:
                if anomaly["severity"] == "critical":
                    alert = SecurityAlert(
                        timestamp=datetime.now(),
                        alert_type=anomaly["type"],
                        severity=anomaly["severity"],
                        user_id=anomaly.get("user_id", "unknown"),
                        description=anomaly["description"],
                        evidence={"anomaly_data": anomaly},
                        recommended_action=anomaly["recommended_action"]
                    )
                    self.security_alerts.append(alert)
            
            return AnomalyReport(
                report_period="1_hour",
                anomalies_detected=len(anomalies),
                anomaly_details=anomalies,
                critical_anomalies=len([a for a in anomalies if a["severity"] == "critical"]),
                high_anomalies=len([a for a in anomalies if a["severity"] == "high"]),
                medium_anomalies=len([a for a in anomalies if a["severity"] == "medium"]),
                recommendations=list(set([a["recommended_action"] for a in anomalies])),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomalous access: {e}")
            return AnomalyReport(
                report_period="error",
                anomalies_detected=0,
                anomaly_details=[],
                critical_anomalies=0,
                high_anomalies=0,
                medium_anomalies=0,
                recommendations=["Error occurred during anomaly detection"],
                timestamp=datetime.now()
            )
    
    async def ensure_data_compliance(self) -> ComplianceReport:
        """
        Ensure data handling compliance with GDPR, CCPA, and other regulations.
        
        Returns:
            ComplianceReport with compliance status and recommendations
        """
        try:
            compliance_checks = {}
            recommendations = []
            
            # GDPR Compliance Checks
            gdpr_checks = {
                "data_encryption": self._check_data_encryption(),
                "access_logging": len(self.access_logs) > 0,
                "user_consent_tracking": self._check_user_consent_tracking(),
                "data_retention_policy": self._check_data_retention_policy(),
                "right_to_deletion": self._check_deletion_capabilities()
            }
            
            gdpr_compliance = all(gdpr_checks.values())
            compliance_checks["GDPR"] = {
                "compliant": gdpr_compliance,
                "checks": gdpr_checks,
                "score": sum(gdpr_checks.values()) / len(gdpr_checks)
            }
            
            if not gdpr_compliance:
                recommendations.extend([
                    "Implement data encryption for all sensitive fields",
                    "Enhance access logging and monitoring",
                    "Implement user consent tracking mechanisms",
                    "Define and implement data retention policies",
                    "Ensure right to deletion capabilities"
                ])
            
            # CCPA Compliance Checks
            ccpa_checks = {
                "data_transparency": self._check_data_transparency(),
                "opt_out_mechanisms": self._check_opt_out_mechanisms(),
                "data_sale_restrictions": self._check_data_sale_restrictions(),
                "consumer_rights": self._check_consumer_rights()
            }
            
            ccpa_compliance = all(ccpa_checks.values())
            compliance_checks["CCPA"] = {
                "compliant": ccpa_compliance,
                "checks": ccpa_checks,
                "score": sum(ccpa_checks.values()) / len(ccpa_checks)
            }
            
            if not ccpa_compliance:
                recommendations.extend([
                    "Implement data transparency mechanisms",
                    "Provide opt-out mechanisms for data processing",
                    "Ensure no unauthorized data sales",
                    "Implement consumer rights management"
                ])
            
            # Overall compliance score
            total_checks = sum(len(checks["checks"]) for checks in compliance_checks.values())
            passed_checks = sum(sum(checks["checks"].values()) for checks in compliance_checks.values())
            overall_score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            return ComplianceReport(
                compliance_standards=list(compliance_checks.keys()),
                overall_compliance_score=overall_score,
                compliance_by_standard=compliance_checks,
                non_compliant_areas=[
                    standard for standard, data in compliance_checks.items()
                    if not data["compliant"]
                ],
                recommendations=list(set(recommendations)),
                last_assessment=datetime.now(),
                next_assessment_due=datetime.now() + timedelta(days=30)
            )
            
        except Exception as e:
            logger.error(f"Error ensuring data compliance: {e}")
            return ComplianceReport(
                compliance_standards=[],
                overall_compliance_score=0.0,
                compliance_by_standard={},
                non_compliant_areas=["Error during compliance assessment"],
                recommendations=["Investigate compliance assessment errors"],
                last_assessment=datetime.now(),
                next_assessment_due=datetime.now() + timedelta(days=1)
            )
    
    def log_access(self, user_id: str, access_type: AccessType, collection_name: str, 
                   operation: str, success: bool = True, error_message: Optional[str] = None,
                   source_ip: Optional[str] = None, data_accessed: Optional[Dict[str, Any]] = None):
        """
        Log database access for audit trail.
        
        Args:
            user_id: User performing the access
            access_type: Type of access operation
            collection_name: Collection being accessed
            operation: Specific operation performed
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            source_ip: Source IP address
            data_accessed: Data that was accessed
        """
        try:
            # Identify sensitive fields in accessed data
            sensitive_fields = []
            if data_accessed:
                for key in data_accessed.keys():
                    if self._is_sensitive_field(key):
                        sensitive_fields.append(key)
            
            # Create access log entry
            log_entry = AccessLogEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                access_type=access_type,
                collection_name=collection_name,
                operation=operation,
                source_ip=source_ip,
                success=success,
                error_message=error_message,
                data_accessed=data_accessed,
                sensitive_fields=sensitive_fields
            )
            
            # Add to access logs
            self.access_logs.append(log_entry)
            
            # Update user access patterns
            if user_id not in self.user_access_patterns:
                self.user_access_patterns[user_id] = []
            self.user_access_patterns[user_id].append(log_entry)
            
            # Update IP access patterns
            if source_ip:
                if source_ip not in self.ip_access_patterns:
                    self.ip_access_patterns[source_ip] = []
                self.ip_access_patterns[source_ip].append(log_entry)
            
            # Maintain log size limits
            if len(self.access_logs) > 10000:
                self.access_logs = self.access_logs[-10000:]
            
            # Log sensitive data access
            if sensitive_fields:
                logger.info(f"Sensitive data access logged: user={user_id}, fields={sensitive_fields}")
            
        except Exception as e:
            logger.error(f"Error logging access: {e}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        try:
            recent_time = datetime.now() - timedelta(hours=24)
            recent_logs = [log for log in self.access_logs if log.timestamp >= recent_time]
            recent_alerts = [alert for alert in self.security_alerts if alert.timestamp >= recent_time]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "security_overview": {
                    "total_accesses_24h": len(recent_logs),
                    "failed_accesses_24h": len([log for log in recent_logs if not log.success]),
                    "unique_users_24h": len(set(log.user_id for log in recent_logs)),
                    "sensitive_data_accesses_24h": len([log for log in recent_logs if log.sensitive_fields]),
                    "security_alerts_24h": len(recent_alerts),
                    "critical_alerts_24h": len([alert for alert in recent_alerts if alert.severity == "critical"])
                },
                "access_patterns": {
                    "top_users": dict(sorted(
                        [(user, len(logs)) for user, logs in self.user_access_patterns.items()],
                        key=lambda x: x[1], reverse=True
                    )[:10]),
                    "access_by_hour": self._get_access_by_hour(recent_logs),
                    "failed_access_patterns": self._get_failed_access_patterns(recent_logs)
                },
                "recent_alerts": [
                    {
                        "timestamp": alert.timestamp.isoformat(),
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "user_id": alert.user_id,
                        "description": alert.description,
                        "recommended_action": alert.recommended_action
                    }
                    for alert in recent_alerts[-10:]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting security dashboard: {e}")
            return {"error": f"Security dashboard error: {str(e)}"}
    
    # Private helper methods
    
    def _initialize_encryption_key(self) -> bytes:
        """Initialize encryption key for sensitive data."""
        # In production, this should come from a secure key management system
        password = os.getenv("ENCRYPTION_PASSWORD", "default_password_change_in_production").encode()
        salt = os.getenv("ENCRYPTION_SALT", "default_salt_change_in_production").encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _is_connection_encrypted(self) -> bool:
        """Check if MongoDB connection is encrypted."""
        # In a real implementation, this would check the actual connection properties
        # For now, we'll assume encryption based on connection string
        return True  # Placeholder - implement actual TLS check
    
    def _is_authentication_enabled(self) -> bool:
        """Check if MongoDB authentication is enabled."""
        # In a real implementation, this would check authentication configuration
        return True  # Placeholder - implement actual auth check
    
    async def _validate_database_permissions(self) -> bool:
        """Validate database permissions are not overly permissive."""
        # In a real implementation, this would check user roles and permissions
        return True  # Placeholder - implement actual permission validation
    
    def _validate_chromadb_security(self) -> bool:
        """Validate ChromaDB security configuration."""
        # In a real implementation, this would check ChromaDB security settings
        return True  # Placeholder - implement actual ChromaDB security check
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field contains sensitive data."""
        field_lower = field_name.lower()
        for pattern in self.sensitive_field_patterns:
            if re.search(pattern, field_lower):
                return True
        return False
    
    def _check_data_encryption(self) -> bool:
        """Check if data encryption is properly implemented."""
        return hasattr(self, 'cipher_suite') and self.cipher_suite is not None
    
    def _check_user_consent_tracking(self) -> bool:
        """Check if user consent tracking is implemented."""
        # In a real implementation, this would check consent management
        return False  # Placeholder - implement consent tracking check
    
    def _check_data_retention_policy(self) -> bool:
        """Check if data retention policy is implemented."""
        # In a real implementation, this would check retention policies
        return False  # Placeholder - implement retention policy check
    
    def _check_deletion_capabilities(self) -> bool:
        """Check if right to deletion is implemented."""
        # In a real implementation, this would check deletion capabilities
        return False  # Placeholder - implement deletion capability check
    
    def _check_data_transparency(self) -> bool:
        """Check if data transparency mechanisms are implemented."""
        return False  # Placeholder - implement transparency check
    
    def _check_opt_out_mechanisms(self) -> bool:
        """Check if opt-out mechanisms are implemented."""
        return False  # Placeholder - implement opt-out check
    
    def _check_data_sale_restrictions(self) -> bool:
        """Check if data sale restrictions are implemented."""
        return True  # Assuming no data sales
    
    def _check_consumer_rights(self) -> bool:
        """Check if consumer rights management is implemented."""
        return False  # Placeholder - implement consumer rights check
    
    def _get_access_by_hour(self, logs: List[AccessLogEntry]) -> Dict[int, int]:
        """Get access counts by hour of day."""
        access_by_hour = {hour: 0 for hour in range(24)}
        for log in logs:
            access_by_hour[log.timestamp.hour] += 1
        return access_by_hour
    
    def _get_failed_access_patterns(self, logs: List[AccessLogEntry]) -> Dict[str, int]:
        """Get patterns of failed access attempts."""
        failed_logs = [log for log in logs if not log.success]
        patterns = {}
        for log in failed_logs:
            key = f"{log.user_id}:{log.access_type.value}"
            patterns[key] = patterns.get(key, 0) + 1
        return patterns


# Global security manager instance
security_compliance_manager = SecurityComplianceManager() 