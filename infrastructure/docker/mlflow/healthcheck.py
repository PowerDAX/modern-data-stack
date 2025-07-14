#!/usr/bin/env python3
"""
MLflow Health Check Script
Comprehensive health validation for production MLflow deployment
"""

import os
import sys
import time
import json
import logging
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import psycopg2
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLflowHealthChecker:
    """Comprehensive MLflow health checker"""
    
    def __init__(self):
        self.mlflow_host = os.getenv('MLFLOW_SERVER_HOST', 'localhost')
        self.mlflow_port = os.getenv('MLFLOW_SERVER_PORT', '5000')
        self.backend_store_uri = os.getenv('MLFLOW_BACKEND_STORE_URI', '')
        self.artifact_root = os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT', '')
        self.s3_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL', '')
        self.timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
        
        self.health_results = {
            'timestamp': time.time(),
            'status': 'unknown',
            'checks': {},
            'version': self._get_mlflow_version()
        }
    
    def _get_mlflow_version(self) -> str:
        """Get MLflow version"""
        try:
            import mlflow
            return mlflow.__version__
        except ImportError:
            return 'unknown'
    
    def check_mlflow_server(self) -> Dict[str, Any]:
        """Check if MLflow server is responding"""
        try:
            url = f"http://{self.mlflow_host}:{self.mlflow_port}/health"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f'HTTP {response.status_code}: {response.text}',
                    'status_code': response.status_code
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_database_connection(self) -> Dict[str, Any]:
        """Check database connectivity"""
        if not self.backend_store_uri.startswith('postgresql'):
            return {
                'status': 'skipped',
                'reason': 'Not using PostgreSQL backend'
            }
        
        try:
            # Parse connection string
            parsed = urlparse(self.backend_store_uri)
            
            # Create connection
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                user=parsed.username,
                password=parsed.password,
                connect_timeout=self.timeout
            )
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                'status': 'healthy',
                'database': parsed.path.lstrip('/'),
                'host': parsed.hostname,
                'port': parsed.port or 5432
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_artifact_storage(self) -> Dict[str, Any]:
        """Check artifact storage accessibility"""
        if self.artifact_root.startswith('s3://'):
            return self._check_s3_storage()
        elif self.artifact_root.startswith('file://'):
            return self._check_file_storage()
        else:
            return {
                'status': 'skipped',
                'reason': 'Unknown artifact storage type'
            }
    
    def _check_s3_storage(self) -> Dict[str, Any]:
        """Check S3 storage accessibility"""
        try:
            # Parse S3 path
            s3_path = self.artifact_root.replace('s3://', '')
            bucket_name = s3_path.split('/')[0]
            
            # Configure S3 client
            s3_config = {}
            if self.s3_endpoint:
                s3_config['endpoint_url'] = self.s3_endpoint
            
            s3_client = boto3.client('s3', **s3_config)
            
            # Test bucket access
            s3_client.head_bucket(Bucket=bucket_name)
            
            # Test write access
            test_key = f"health-check-{int(time.time())}.txt"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=test_key,
                Body=b'health check test'
            )
            
            # Test read access
            s3_client.get_object(Bucket=bucket_name, Key=test_key)
            
            # Clean up test object
            s3_client.delete_object(Bucket=bucket_name, Key=test_key)
            
            return {
                'status': 'healthy',
                'bucket': bucket_name,
                'endpoint': self.s3_endpoint or 'default'
            }
            
        except NoCredentialsError:
            return {
                'status': 'unhealthy',
                'error': 'No AWS credentials configured'
            }
        except ClientError as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _check_file_storage(self) -> Dict[str, Any]:
        """Check file storage accessibility"""
        try:
            # Parse file path
            file_path = self.artifact_root.replace('file://', '')
            
            # Check if directory exists
            if not os.path.exists(file_path):
                return {
                    'status': 'unhealthy',
                    'error': f'Directory does not exist: {file_path}'
                }
            
            # Check if directory is writable
            test_file = os.path.join(file_path, f'health-check-{int(time.time())}.txt')
            
            try:
                with open(test_file, 'w') as f:
                    f.write('health check test')
                
                # Test read access
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Clean up test file
                os.remove(test_file)
                
                return {
                    'status': 'healthy',
                    'path': file_path,
                    'writable': True,
                    'readable': True
                }
                
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'error': f'File access error: {str(e)}'
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
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            import shutil
            
            # Check disk usage of artifact directory
            if self.artifact_root.startswith('file://'):
                file_path = self.artifact_root.replace('file://', '')
                total, used, free = shutil.disk_usage(file_path)
                
                usage_percent = (used / total) * 100
                
                return {
                    'status': 'healthy',
                    'disk_usage_percent': usage_percent,
                    'disk_free_mb': free / (1024 * 1024),
                    'disk_total_mb': total / (1024 * 1024),
                    'path': file_path
                }
            else:
                return {
                    'status': 'skipped',
                    'reason': 'Not using file storage'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        logger.info("Starting MLflow health checks...")
        
        # Run individual checks
        checks = {
            'mlflow_server': self.check_mlflow_server(),
            'database': self.check_database_connection(),
            'artifact_storage': self.check_artifact_storage(),
            'memory': self.check_memory_usage(),
            'disk': self.check_disk_usage()
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
    checker = MLflowHealthChecker()
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