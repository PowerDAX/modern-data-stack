# DevOps-Focused Jupyter Startup Script
# Sets up DevOps environment and common utilities

import os
import sys
import warnings
import subprocess
import json

# Suppress warnings in production
if os.environ.get('JUPYTER_ENV') == 'production':
    warnings.filterwarnings('ignore')

# Set up common DevOps imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# DevOps libraries
try:
    import requests
    import yaml
    import json
    print("✅ Core DevOps libraries loaded")
except ImportError:
    print("❌ Some DevOps libraries not available")

try:
    import kubernetes
    from kubernetes import client, config
    print(f"✅ Kubernetes client loaded")
except ImportError:
    print("❌ Kubernetes client not available")

try:
    import docker
    print(f"✅ Docker client loaded")
except ImportError:
    print("❌ Docker client not available")

try:
    import boto3
    print(f"✅ AWS SDK loaded")
except ImportError:
    print("❌ AWS SDK not available")

try:
    import psutil
    print(f"✅ System monitoring (psutil) loaded")
except ImportError:
    print("❌ System monitoring not available")

# Configure matplotlib for inline plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('seaborn-v0_8')

# Set up pandas display options for logs and metrics
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Set up plotting style
sns.set_palette("husl")

# DevOps utility functions
def run_command(command, capture_output=True, shell=True):
    """Execute shell command and return result"""
    try:
        result = subprocess.run(command, 
                              shell=shell, 
                              capture_output=capture_output, 
                              text=True, 
                              timeout=30)
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def check_service_health(service_name, port=None):
    """Check if a service is healthy"""
    if port:
        result = run_command(f"curl -f http://{service_name}:{port}/health")
        return result['success']
    else:
        result = run_command(f"docker ps | grep {service_name}")
        return result['success']

def get_system_info():
    """Get comprehensive system information"""
    info = {
        'cpu_count': os.cpu_count(),
        'memory_total': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available': round(psutil.virtual_memory().available / (1024**3), 2),
        'disk_usage': round(psutil.disk_usage('/').percent, 2),
        'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else 'N/A'
    }
    return info

def check_docker_status():
    """Check Docker daemon status and container information"""
    try:
        if 'docker' in sys.modules:
            client = docker.from_env()
            containers = client.containers.list(all=True)
            return {
                'daemon_running': True,
                'total_containers': len(containers),
                'running_containers': len([c for c in containers if c.status == 'running']),
                'containers': [{'name': c.name, 'status': c.status} for c in containers]
            }
        else:
            result = run_command("docker ps")
            return {
                'daemon_running': result['success'],
                'containers': result['stdout'].split('\n')[1:] if result['success'] else []
            }
    except Exception as e:
        return {'daemon_running': False, 'error': str(e)}

def check_kubernetes_status():
    """Check Kubernetes cluster status"""
    try:
        result = run_command("kubectl cluster-info")
        if result['success']:
            nodes_result = run_command("kubectl get nodes --no-headers")
            return {
                'cluster_accessible': True,
                'cluster_info': result['stdout'],
                'nodes': nodes_result['stdout'].split('\n') if nodes_result['success'] else []
            }
        else:
            return {'cluster_accessible': False, 'error': result['stderr']}
    except Exception as e:
        return {'cluster_accessible': False, 'error': str(e)}

def monitor_resources():
    """Monitor system resources and return as DataFrame"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        data = {
            'metric': ['CPU Usage %', 'Memory Usage %', 'Disk Usage %'],
            'value': [cpu_percent, memory.percent, disk.percent],
            'status': [
                'OK' if cpu_percent < 80 else 'HIGH',
                'OK' if memory.percent < 80 else 'HIGH',
                'OK' if disk.percent < 80 else 'HIGH'
            ]
        }
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error monitoring resources: {e}")
        return pd.DataFrame()

def parse_logs(log_text, log_format='json'):
    """Parse log text and return structured data"""
    lines = log_text.strip().split('\n')
    parsed_logs = []
    
    for line in lines:
        if log_format == 'json':
            try:
                parsed_logs.append(json.loads(line))
            except json.JSONDecodeError:
                parsed_logs.append({'raw': line})
        else:
            parsed_logs.append({'raw': line})
    
    return pd.DataFrame(parsed_logs)

# Set up environment variables for DevOps workflows
os.environ['PYTHONPATH'] = f"/home/jovyan/work:{os.environ.get('PYTHONPATH', '')}"

# Check tool availability
def check_devops_tools():
    """Check availability of common DevOps tools"""
    tools = {
        'docker': 'docker --version',
        'kubectl': 'kubectl version --client',
        'helm': 'helm version',
        'terraform': 'terraform version',
        'aws': 'aws --version',
        'azure': 'az --version',
        'gcloud': 'gcloud version'
    }
    
    available_tools = {}
    for tool, command in tools.items():
        result = run_command(command)
        available_tools[tool] = {
            'available': result['success'],
            'version': result['stdout'].split('\n')[0] if result['success'] else 'Not available'
        }
    
    return available_tools

# Display environment information
print("🚀 DevOps Jupyter environment initialized successfully!")
print(f"📊 Pandas {pd.__version__} | NumPy {np.__version__}")
print(f"🔍 Current working directory: {os.getcwd()}")
print(f"🧠 Python version: {sys.version}")

# Display system information
sys_info = get_system_info()
print(f"💻 System Info:")
print(f"   - CPU Cores: {sys_info['cpu_count']}")
print(f"   - Memory: {sys_info['memory_available']:.1f}GB / {sys_info['memory_total']:.1f}GB")
print(f"   - Disk Usage: {sys_info['disk_usage']:.1f}%")

# Check DevOps tools
print(f"\n🛠️  DevOps Tools Status:")
tools_status = check_devops_tools()
for tool, status in tools_status.items():
    emoji = "✅" if status['available'] else "❌"
    print(f"   {emoji} {tool}: {status['version']}")

# Check Docker status
docker_status = check_docker_status()
if docker_status['daemon_running']:
    print(f"\n🐳 Docker: {docker_status.get('total_containers', 0)} containers ({docker_status.get('running_containers', 0)} running)")
else:
    print(f"\n🐳 Docker: Not accessible")

# Check Kubernetes status
k8s_status = check_kubernetes_status()
if k8s_status['cluster_accessible']:
    print(f"\n⚓ Kubernetes: Cluster accessible ({len(k8s_status['nodes'])} nodes)")
else:
    print(f"\n⚓ Kubernetes: Not accessible")

print("\n🛠️  Available utility functions:")
print("   - run_command(cmd): Execute shell commands")
print("   - check_service_health(service, port): Check service health")
print("   - get_system_info(): Get system information")
print("   - monitor_resources(): Monitor system resources")
print("   - check_docker_status(): Check Docker status")
print("   - check_kubernetes_status(): Check Kubernetes status")
print("   - parse_logs(text, format): Parse log files")
print("\n🎯 Ready for DevOps automation workflows!") 