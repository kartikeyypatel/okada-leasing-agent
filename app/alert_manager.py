# /app/alert_manager.py
"""
Automated Alerting and Notification System

This module provides comprehensive alerting and notification capabilities
for MongoDB and RAG system monitoring, with configurable thresholds,
multiple notification channels, and intelligent alert management.
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import os

from app.mongodb_health_monitor import mongodb_health_monitor
from app.chromadb_performance_optimizer import chromadb_performance_optimizer
from app.concurrent_operations_manager import concurrent_operations_manager
from app.security_compliance_manager import security_compliance_manager
from app.mongodb_error_handler import mongodb_error_handler

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONNECTION_FAILURE = "connection_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_BREACH = "security_breach"
    DATA_CORRUPTION = "data_corruption"
    SERVICE_UNAVAILABLE = "service_unavailable"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    SYSTEM_ERROR = "system_error"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    SMS = "sms"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    evaluation_window_minutes: int
    notification_channels: List[NotificationChannel]
    enabled: bool = True
    cooldown_minutes: int = 15
    auto_resolve: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Individual alert instance."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notification_sent: bool = False
    auto_resolved: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationTemplate:
    """Template for alert notifications."""
    template_id: str
    channel: NotificationChannel
    severity: AlertSeverity
    subject_template: str
    body_template: str
    format_type: str = "text"  # text, html, json


class AlertManager:
    """
    Comprehensive automated alerting and notification system.
    
    Provides configurable alert rules, multiple notification channels,
    intelligent alert management, and automated escalation.
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_templates: Dict[str, NotificationTemplate] = {}
        
        # Alert management settings
        self.max_alerts_per_rule = 10
        self.alert_history_limit = 1000
        self.default_cooldown_minutes = 15
        self.escalation_timeout_minutes = 60
        
        # Notification settings
        self.notification_config = {
            "email": {
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME", ""),
                "password": os.getenv("SMTP_PASSWORD", ""),
                "from_address": os.getenv("ALERT_FROM_EMAIL", "alerts@okadaleasing.com"),
                "to_addresses": os.getenv("ALERT_TO_EMAILS", "admin@okadaleasing.com").split(",")
            },
            "slack": {
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
                "channel": os.getenv("SLACK_CHANNEL", "#alerts"),
                "username": "MongoDB Alert Bot"
            },
            "webhook": {
                "url": os.getenv("WEBHOOK_URL", ""),
                "headers": {"Content-Type": "application/json"},
                "timeout": 10
            }
        }
        
        # Background monitoring
        self._monitoring_task = None
        self._monitoring_interval = 60  # seconds
        
        # Initialize default alert rules and templates
        self._initialize_default_rules()
        self._initialize_notification_templates()
        
        logger.info("Alert Manager initialized with default rules and templates")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Alert rule added/updated: {rule.name} ({rule.rule_id})")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Alert rule removed: {rule_id}")
    
    def enable_alert_rule(self, rule_id: str, enabled: bool = True):
        """Enable or disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Alert rule {rule_id} {status}")
    
    async def evaluate_alerts(self) -> List[Alert]:
        """Evaluate all alert rules and trigger new alerts if necessary."""
        new_alerts = []
        
        try:
            # Get current system metrics
            metrics = await self._collect_system_metrics()
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Check if rule is in cooldown
                    if self._is_rule_in_cooldown(rule_id):
                        continue
                    
                    # Evaluate rule condition
                    alert_triggered = await self._evaluate_rule_condition(rule, metrics)
                    
                    if alert_triggered:
                        # Create new alert
                        alert = await self._create_alert(rule, metrics)
                        new_alerts.append(alert)
                        
                        # Add to active alerts
                        self.active_alerts[alert.alert_id] = alert
                        
                        # Send notifications
                        await self._send_alert_notifications(alert)
                        
                        logger.warning(f"Alert triggered: {alert.title}")
                    
                except Exception as rule_error:
                    logger.error(f"Error evaluating alert rule {rule_id}: {rule_error}")
                    continue
            
            # Check for auto-resolution of existing alerts
            await self._check_alert_auto_resolution(metrics)
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, auto_resolved: bool = False) -> bool:
        """Resolve an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                alert.auto_resolved = auto_resolved
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                # Send resolution notification if configured
                await self._send_resolution_notification(alert)
                
                logger.info(f"Alert resolved: {alert_id} (auto: {auto_resolved})")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def start_monitoring(self):
        """Start background alert monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Alert monitoring started")
    
    async def stop_monitoring(self):
        """Stop background alert monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Alert monitoring stopped")
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive alert dashboard data."""
        try:
            recent_time = datetime.now() - timedelta(hours=24)
            recent_alerts = [alert for alert in self.alert_history if alert.triggered_at >= recent_time]
            
            # Count alerts by severity
            severity_counts = {}
            for severity in AlertSeverity:
                count = len([alert for alert in recent_alerts if alert.severity == severity])
                severity_counts[severity.value] = count
            
            # Count alerts by type
            type_counts = {}
            for alert_type in AlertType:
                count = len([alert for alert in recent_alerts if alert.alert_type == alert_type])
                type_counts[alert_type.value] = count
            
            # Calculate metrics
            total_alerts_24h = len(recent_alerts)
            active_alerts_count = len(self.active_alerts)
            acknowledged_alerts = len([alert for alert in self.active_alerts.values() if alert.acknowledged_at])
            
            # Alert resolution metrics
            resolved_alerts = [alert for alert in recent_alerts if alert.resolved_at]
            avg_resolution_time = 0.0
            if resolved_alerts:
                resolution_times = [
                    (alert.resolved_at - alert.triggered_at).total_seconds() / 60
                    for alert in resolved_alerts if alert.resolved_at
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "alert_summary": {
                    "active_alerts": active_alerts_count,
                    "acknowledged_alerts": acknowledged_alerts,
                    "total_alerts_24h": total_alerts_24h,
                    "avg_resolution_time_minutes": avg_resolution_time,
                    "alert_rules_enabled": len([rule for rule in self.alert_rules.values() if rule.enabled])
                },
                "alerts_by_severity": severity_counts,
                "alerts_by_type": type_counts,
                "active_alerts_detail": [
                    {
                        "alert_id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "type": alert.alert_type.value,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "acknowledged": alert.acknowledged_at is not None,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value
                    }
                    for alert in self.active_alerts.values()
                ],
                "recent_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                        "auto_resolved": alert.auto_resolved
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting alert dashboard: {e}")
            return {"error": f"Dashboard error: {str(e)}"}
    
    # Private methods
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await self.evaluate_alerts()
                await asyncio.sleep(self._monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._monitoring_interval)
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for alert evaluation."""
        metrics = {}
        
        try:
            # MongoDB health metrics
            db_status = await mongodb_health_monitor.check_connection_health()
            metrics["mongodb_ping_time_ms"] = db_status.ping_time_ms
            metrics["mongodb_connection_available"] = 1.0 if db_status.connection_available else 0.0
            
            # Query performance metrics
            query_performance = await mongodb_health_monitor.analyze_query_performance()
            metrics["avg_query_duration_ms"] = query_performance.avg_duration_ms
            metrics["slow_queries_count"] = float(query_performance.slow_queries_count)
            
            # Connection pool metrics
            pool_status = await concurrent_operations_manager.manage_connection_pool()
            metrics["connection_pool_utilization"] = pool_status.pool_utilization
            metrics["waiting_connections"] = float(pool_status.waiting_connections)
            
            # Resource usage metrics
            resource_report = await concurrent_operations_manager.monitor_resource_usage()
            metrics["cpu_usage_percent"] = resource_report.cpu_usage_percent
            metrics["memory_usage_mb"] = resource_report.memory_usage_mb
            
            # Security metrics
            access_audit = await security_compliance_manager.audit_access_patterns()
            metrics["failed_accesses_rate"] = (access_audit.failed_accesses / max(access_audit.total_accesses, 1)) * 100
            
            # Error metrics
            error_stats = mongodb_error_handler.get_error_statistics()
            metrics["error_count_24h"] = float(error_stats.get("error_summary", {}).get("total_errors_24h", 0))
            metrics["connection_failure_rate"] = float(error_stats.get("connection_health", {}).get("consecutive_failures", 0))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def _evaluate_rule_condition(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """Evaluate if an alert rule condition is met."""
        try:
            if rule.metric_name not in metrics:
                return False
            
            current_value = metrics[rule.metric_name]
            threshold = rule.threshold_value
            operator = rule.comparison_operator
            
            if operator == ">":
                return current_value > threshold
            elif operator == "<":
                return current_value < threshold
            elif operator == ">=":
                return current_value >= threshold
            elif operator == "<=":
                return current_value <= threshold
            elif operator == "==":
                return current_value == threshold
            elif operator == "!=":
                return current_value != threshold
            else:
                logger.warning(f"Unknown comparison operator: {operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    async def _create_alert(self, rule: AlertRule, metrics: Dict[str, float]) -> Alert:
        """Create a new alert from a triggered rule."""
        alert_id = f"{rule.rule_id}_{int(datetime.now().timestamp())}"
        current_value = metrics.get(rule.metric_name, 0.0)
        
        # Generate alert message
        message = f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold_value}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.name,
            message=message,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            triggered_at=datetime.now(),
            context={"metrics": metrics},
            tags=rule.tags
        )
        
        return alert
    
    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if a rule is in cooldown period."""
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        # Find the most recent alert for this rule
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.rule_id == rule_id and 
            datetime.now() - alert.triggered_at < timedelta(minutes=rule.cooldown_minutes)
        ]
        
        return len(recent_alerts) > 0
    
    async def _check_alert_auto_resolution(self, metrics: Dict[str, float]):
        """Check for auto-resolution of existing alerts."""
        to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.auto_resolve:
                continue
            
            # Check if the condition is no longer met
            condition_met = await self._evaluate_rule_condition(rule, metrics)
            if not condition_met:
                to_resolve.append(alert_id)
        
        # Resolve alerts
        for alert_id in to_resolve:
            await self.resolve_alert(alert_id, auto_resolved=True)
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for a new alert."""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
        
        for channel in rule.notification_channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(alert)
                elif channel == NotificationChannel.LOG:
                    await self._send_log_notification(alert)
                
            except Exception as e:
                logger.error(f"Error sending {channel.value} notification for alert {alert.alert_id}: {e}")
        
        alert.notification_sent = True
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification for an alert."""
        try:
            config = self.notification_config["email"]
            if not config["username"] or not config["to_addresses"]:
                logger.warning("Email configuration incomplete - skipping email notification")
                return
            
            # Create email content
            subject = f"[{alert.severity.value.upper()}] {alert.title}"
            body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Severity: {alert.severity.value}
- Type: {alert.alert_type.value}
- Message: {alert.message}
- Triggered: {alert.triggered_at.isoformat()}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}

Please investigate and take appropriate action.
            """.strip()
            
            # Send email
            msg = MIMEMultipart()
            msg["From"] = config["from_address"]
            msg["To"] = ", ".join(config["to_addresses"])
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
                server.starttls()
                if config["username"] and config["password"]:
                    server.login(config["username"], config["password"])
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification for an alert."""
        try:
            config = self.notification_config["slack"]
            if not config["webhook_url"]:
                logger.warning("Slack webhook URL not configured - skipping Slack notification")
                return
            
            # Create Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }.get(alert.severity, "warning")
            
            payload = {
                "channel": config["channel"],
                "username": config["username"],
                "attachments": [{
                    "color": color,
                    "title": f"{alert.severity.value.upper()}: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                        {"title": "Triggered", "value": alert.triggered_at.isoformat(), "short": True}
                    ],
                    "footer": "MongoDB Alert System",
                    "ts": int(alert.triggered_at.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config["webhook_url"], json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.alert_id}")
                    else:
                        logger.error(f"Slack notification failed with status {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification for an alert."""
        try:
            config = self.notification_config["webhook"]
            if not config["url"]:
                logger.warning("Webhook URL not configured - skipping webhook notification")
                return
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.value,
                "type": alert.alert_type.value,
                "title": alert.title,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "tags": alert.tags
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["url"],
                    json=payload,
                    headers=config["headers"],
                    timeout=aiohttp.ClientTimeout(total=config["timeout"])
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                    else:
                        logger.error(f"Webhook notification failed with status {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_log_notification(self, alert: Alert):
        """Send log notification for an alert."""
        log_message = f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        
        if alert.severity == AlertSeverity.CRITICAL or alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send notification when an alert is resolved."""
        # Similar to alert notifications but for resolution
        # Implementation would be similar to _send_alert_notifications
        logger.info(f"Alert resolved: {alert.title}")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="mongodb_connection_failure",
                name="MongoDB Connection Failure",
                description="MongoDB connection is not available",
                alert_type=AlertType.CONNECTION_FAILURE,
                severity=AlertSeverity.CRITICAL,
                metric_name="mongodb_connection_available",
                threshold_value=0.5,
                comparison_operator="<",
                evaluation_window_minutes=5,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG],
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="high_query_latency",
                name="High Query Latency",
                description="Average query duration is too high",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.WARNING,
                metric_name="avg_query_duration_ms",
                threshold_value=1000.0,
                comparison_operator=">",
                evaluation_window_minutes=10,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG],
                auto_resolve=True
            ),
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="System CPU usage is critically high",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.CRITICAL,
                metric_name="cpu_usage_percent",
                threshold_value=90.0,
                comparison_operator=">",
                evaluation_window_minutes=5,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG],
                auto_resolve=True
            ),
            AlertRule(
                rule_id="connection_pool_exhaustion",
                name="Connection Pool Exhaustion",
                description="Connection pool utilization is too high",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.WARNING,
                metric_name="connection_pool_utilization",
                threshold_value=0.9,
                comparison_operator=">",
                evaluation_window_minutes=5,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG],
                auto_resolve=True
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Too many errors in the last 24 hours",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.WARNING,
                metric_name="error_count_24h",
                threshold_value=50.0,
                comparison_operator=">",
                evaluation_window_minutes=15,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG],
                cooldown_minutes=30
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _initialize_notification_templates(self):
        """Initialize notification templates."""
        # Email templates
        self.notification_templates["email_critical"] = NotificationTemplate(
            template_id="email_critical",
            channel=NotificationChannel.EMAIL,
            severity=AlertSeverity.CRITICAL,
            subject_template="[CRITICAL] MongoDB Alert: {title}",
            body_template="""
CRITICAL ALERT

Alert: {title}
Severity: {severity}
Message: {message}
Triggered: {triggered_at}

Metric: {metric_name}
Current Value: {current_value}
Threshold: {threshold_value}

Immediate action required!
            """.strip()
        )
        
        # Add more templates as needed
        logger.info("Notification templates initialized")


# Global alert manager instance
alert_manager = AlertManager() 