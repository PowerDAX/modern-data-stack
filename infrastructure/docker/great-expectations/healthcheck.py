#!/usr/bin/env python3
"""
Great Expectations Health Check Script
Comprehensive health validation for production Great Expectations deployment
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GreatExpectationsHealthChecker:
    """Comprehensive Great Expectations health checker"""
    
    def __init__(self):
        self.ge_home = os.getenv('GE_HOME', '/opt/great_expectations')
        self.data_docs_port = int(os.getenv('DATA_DOCS_PORT', '8082'))
        self.timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
        
        self.health_results = {
            'timestamp': time.time(),
            'status': 'unknown',
            'checks': {},
            'version': self._get_ge_version()
        }
    
    def _get_ge_version(self) -> str:
        """Get Great Expectations version"""
        try:
            import great_expectations
            return great_expectations.__version__
        except ImportError:
            return 'unknown'
    
    def check_data_context(self) -> Dict[str, Any]:
        """Check if Great Expectations data context is valid"""
        try:
            from great_expectations.data_context import DataContext
            
            # Load data context
            context = DataContext(self.ge_home)
            
            # Get basic context info
            project_config = context.get_config()
            
            # Check if context is valid
            if context and project_config:
                return {
                    'status': 'healthy',
                    'config_version': project_config.config_version,
                    'datasources_count': len(project_config.datasources),
                    'stores_count': len(project_config.stores),
                    'data_docs_sites_count': len(project_config.data_docs_sites)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Invalid data context'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_expectation_suites(self) -> Dict[str, Any]:
        """Check expectation suites availability"""
        try:
            from great_expectations.data_context import DataContext
            
            context = DataContext(self.ge_home)
            
            # List expectation suites
            expectation_suites = context.list_expectation_suites()
            
            # Check if suites directory exists
            expectations_dir = Path(self.ge_home) / 'expectations'
            
            return {
                'status': 'healthy',
                'expectation_suites_count': len(expectation_suites),
                'expectation_suites': expectation_suites,
                'expectations_directory_exists': expectations_dir.exists()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_data_docs(self) -> Dict[str, Any]:
        """Check data docs availability"""
        try:
            # Check if data docs server is running
            try:
                url = f"http://localhost:{self.data_docs_port}"
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    server_status = 'healthy'
                    server_error = None
                else:
                    server_status = 'unhealthy'
                    server_error = f'HTTP {response.status_code}'
            except requests.exceptions.RequestException as e:
                server_status = 'unhealthy'
                server_error = str(e)
            
            # Check if data docs directory exists
            data_docs_dir = Path(self.ge_home) / 'data_docs'
            
            # Check if index.html exists
            index_file = data_docs_dir / 'index.html'
            
            return {
                'status': server_status,
                'server_error': server_error,
                'data_docs_directory_exists': data_docs_dir.exists(),
                'index_file_exists': index_file.exists(),
                'port': self.data_docs_port
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_validations(self) -> Dict[str, Any]:
        """Check validation results"""
        try:
            from great_expectations.data_context import DataContext
            
            context = DataContext(self.ge_home)
            
            # Get validation results
            validation_results = context.get_validation_results()
            
            # Calculate metrics
            total_validations = len(validation_results)
            successful_validations = sum(1 for result in validation_results if result.success)
            failed_validations = total_validations - successful_validations
            
            # Check validations directory
            validations_dir = Path(self.ge_home) / 'validations'
            
            return {
                'status': 'healthy',
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
                'validations_directory_exists': validations_dir.exists()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_checkpoints(self) -> Dict[str, Any]:
        """Check checkpoints availability"""
        try:
            from great_expectations.data_context import DataContext
            
            context = DataContext(self.ge_home)
            
            # List checkpoints
            checkpoints = context.list_checkpoints()
            
            # Check checkpoints directory
            checkpoints_dir = Path(self.ge_home) / 'checkpoints'
            
            return {
                'status': 'healthy',
                'checkpoints_count': len(checkpoints),
                'checkpoints': checkpoints,
                'checkpoints_directory_exists': checkpoints_dir.exists()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            import shutil
            
            # Check disk usage of GE home directory
            total, used, free = shutil.disk_usage(self.ge_home)
            
            usage_percent = (used / total) * 100
            
            return {
                'status': 'healthy',
                'disk_usage_percent': usage_percent,
                'disk_free_mb': free / (1024 * 1024),
                'disk_total_mb': total / (1024 * 1024),
                'path': self.ge_home
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions"""
        try:
            ge_home_path = Path(self.ge_home)
            
            # Check if directories are readable/writable
            directories_to_check = [
                'expectations',
                'validations',
                'checkpoints',
                'data_docs',
                'logs',
                'data'
            ]
            
            permission_results = {}
            
            for dir_name in directories_to_check:
                dir_path = ge_home_path / dir_name
                
                if dir_path.exists():
                    permission_results[dir_name] = {
                        'exists': True,
                        'readable': os.access(dir_path, os.R_OK),
                        'writable': os.access(dir_path, os.W_OK)
                    }
                else:
                    permission_results[dir_name] = {
                        'exists': False,
                        'readable': False,
                        'writable': False
                    }
            
            return {
                'status': 'healthy',
                'directories': permission_results
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            return {
                'status': 'healthy',
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'memory_total_mb': memory.total / (1024 * 1024)
            }
            
        except ImportError:
            return {
                'status': 'skipped',
                'reason': 'psutil not available'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        logger.info("Starting Great Expectations health checks...")
        
        # Run individual checks
        checks = {
            'data_context': self.check_data_context(),
            'expectation_suites': self.check_expectation_suites(),
            'data_docs': self.check_data_docs(),
            'validations': self.check_validations(),
            'checkpoints': self.check_checkpoints(),
            'disk_usage': self.check_disk_usage(),
            'file_permissions': self.check_file_permissions(),
            'memory_usage': self.check_memory_usage()
        }
        
        # Determine overall status
        unhealthy_checks = [
            name for name, result in checks.items()
            if result.get('status') == 'unhealthy'
        ]
        
        if unhealthy_checks:
            overall_status = 'unhealthy'
            logger.error(f"Health checks failed: {', '.join(unhealthy_checks)}")
        else:
            overall_status = 'healthy'
            logger.info("All health checks passed")
        
        self.health_results.update({
            'status': overall_status,
            'checks': checks,
            'failed_checks': unhealthy_checks
        })
        
        return self.health_results
    
    def print_health_status(self, results: Dict[str, Any]):
        """Print health status to stdout"""
        print(json.dumps(results, indent=2))

def main():
    """Main health check function"""
    checker = GreatExpectationsHealthChecker()
    results = checker.run_health_checks()
    
    # Print results
    checker.print_health_status(results)
    
    # Exit with appropriate code
    if results['status'] == 'healthy':
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main() 