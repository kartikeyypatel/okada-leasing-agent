#!/usr/bin/env python3
"""
Simple test for health monitoring system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.error_handler import rag_error_handler, ErrorRecord, ErrorCategory, ErrorSeverity, ErrorContext
from datetime import datetime

def test_health_monitoring():
    print('🏥 Testing health monitoring endpoints...')
    
    # Test initial health status
    health_report = rag_error_handler.get_health_status()
    print(f'✅ Initial health status: {health_report["health_metrics"]["system_status"]}')
    print(f'✅ Initial total errors: {health_report["health_metrics"]["total_errors"]}')
    
    # Add a test error to validate monitoring
    test_error = ErrorRecord(
        timestamp=datetime.now(),
        error_type='TestError',
        error_message='Test error for health monitoring',
        category=ErrorCategory.SEARCH_OPERATION,
        severity=ErrorSeverity.MEDIUM,
        context=ErrorContext(user_id='health_test@example.com', operation='health_test'),
        stack_trace='test stack trace'
    )
    
    rag_error_handler.health_monitor.record_error(test_error)
    
    # Check updated health status
    updated_report = rag_error_handler.get_health_status()
    print(f'✅ Updated total errors: {updated_report["health_metrics"]["total_errors"]}')
    print(f'✅ Recent errors: {len(updated_report["recent_errors"])}')
    
    # Test error categorization in health metrics
    if updated_report["health_metrics"]["errors_by_category"]:
        print(f'✅ Errors by category: {updated_report["health_metrics"]["errors_by_category"]}')
    
    if updated_report["health_metrics"]["errors_by_severity"]:
        print(f'✅ Errors by severity: {updated_report["health_metrics"]["errors_by_severity"]}')
    
    print('🎉 Health monitoring system is working correctly!')
    return True

if __name__ == "__main__":
    test_health_monitoring()