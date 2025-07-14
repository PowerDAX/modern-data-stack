# Advanced Jupyter Notebook Configuration
# Optimized for productivity, security, and performance

import os
from jupyter_core.paths import jupyter_data_dir

# =============================================================================
# Application(SingletonConfigurable) configuration
# =============================================================================

# The IP address the notebook server will listen on.
c.ServerApp.ip = '0.0.0.0'

# The port the notebook server will listen on.
c.ServerApp.port = 8888

# Whether to open in a browser
c.ServerApp.open_browser = False

# Allow root user to run the notebook server
c.ServerApp.allow_root = True

# =============================================================================
# Security Configuration
# =============================================================================

# Disable cross-site-request-forgery protection
c.ServerApp.disable_check_xsrf = True

# Allow all origins for CORS
c.ServerApp.allow_origin = '*'

# Allow remote access
c.ServerApp.allow_remote_access = True

# Enable password authentication (set via environment variable)
if os.environ.get('JUPYTER_PASSWORD'):
    from notebook.auth import passwd
    c.ServerApp.password = passwd(os.environ.get('JUPYTER_PASSWORD'))

# =============================================================================
# Performance Configuration
# =============================================================================

# Increase the maximum size of a message
c.ServerApp.max_message_size = 67108864  # 64MB

# Increase the maximum size of a single uploaded file
c.ServerApp.max_body_size = 268435456  # 256MB

# Enable autosave
c.FileContentsManager.use_atomic_writing = True

# Set autosave interval (in seconds)
c.ServerApp.autosave_interval = 120

# =============================================================================
# Kernel Configuration
# =============================================================================

# Timeout for kernel startup
c.KernelManager.kernel_startup_timeout = 300

# Timeout for kernel shutdown
c.KernelManager.kernel_shutdown_timeout = 60

# Allow kernels to be interrupted
c.KernelManager.interrupt_timeout = 30

# =============================================================================
# File Management Configuration
# =============================================================================

# The directory to use for notebooks and kernels
c.ServerApp.notebook_dir = '/home/jovyan/work'

# Whether to allow hidden files to be served
c.ContentsManager.allow_hidden = True

# Maximum number of files to return in a directory listing
c.ContentsManager.max_copy_files = 100

# =============================================================================
# Extension Configuration
# =============================================================================

# Enable JupyterLab extensions
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True,
    'jupyterlab_git': True,
    'jupyter_ai': True,
}

# =============================================================================
# Logging Configuration
# =============================================================================

# Set log level
c.ServerApp.log_level = 'INFO'

# Enable logging to file
c.ServerApp.log_file = '/home/jovyan/.jupyter/jupyter.log'

# Log format
c.ServerApp.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# Development Configuration
# =============================================================================

# Enable debug mode
c.ServerApp.debug = False

# Enable development mode for extensions
c.LabApp.dev_mode = False

# =============================================================================
# Custom Configuration
# =============================================================================

# Custom CSS for JupyterLab
c.ServerApp.extra_static_paths = ['/home/jovyan/.jupyter/custom']

# Custom JavaScript for JupyterLab
c.ServerApp.extra_template_paths = ['/home/jovyan/.jupyter/templates']

# =============================================================================
# Environment-Specific Configuration
# =============================================================================

# Configuration for different environments
if os.environ.get('JUPYTER_ENV') == 'production':
    # Production-specific settings
    c.ServerApp.token = os.environ.get('JUPYTER_TOKEN', '')
    c.ServerApp.password = os.environ.get('JUPYTER_PASSWORD', '')
    c.ServerApp.allow_origin_pat = r'https://.*\.example\.com'
    c.ServerApp.log_level = 'WARNING'
    
elif os.environ.get('JUPYTER_ENV') == 'development':
    # Development-specific settings
    c.ServerApp.token = ''
    c.ServerApp.password = ''
    c.ServerApp.log_level = 'DEBUG'
    c.LabApp.dev_mode = True
    
else:
    # Default settings for local development
    c.ServerApp.token = ''
    c.ServerApp.password = ''
    c.ServerApp.log_level = 'INFO'

# =============================================================================
# Resource Management
# =============================================================================

# Memory and CPU limits (if running in containerized environment)
if os.environ.get('JUPYTER_MEMORY_LIMIT'):
    c.ResourceUseDisplay.mem_limit = int(os.environ.get('JUPYTER_MEMORY_LIMIT'))

if os.environ.get('JUPYTER_CPU_LIMIT'):
    c.ResourceUseDisplay.cpu_limit = float(os.environ.get('JUPYTER_CPU_LIMIT'))

# =============================================================================
# Collaborative Features
# =============================================================================

# Enable real-time collaboration
c.LabApp.collaborative = True

# Set collaboration room ID
if os.environ.get('JUPYTER_COLLABORATION_ROOM'):
    c.LabApp.collaboration_room = os.environ.get('JUPYTER_COLLABORATION_ROOM')

# =============================================================================
# Database Configuration
# =============================================================================

# Database connection settings (if using database backend)
if os.environ.get('JUPYTER_DATABASE_URL'):
    c.ServerApp.contents_manager_class = 'jupyter_server_database.contents.DatabaseContentsManager'
    c.DatabaseContentsManager.database_url = os.environ.get('JUPYTER_DATABASE_URL')

# =============================================================================
# Backup Configuration
# =============================================================================

# Enable automatic backups
c.FileContentsManager.use_atomic_writing = True

# Set backup directory
if os.environ.get('JUPYTER_BACKUP_DIR'):
    c.FileContentsManager.backup_dir = os.environ.get('JUPYTER_BACKUP_DIR')

# =============================================================================
# Integration Configuration
# =============================================================================

# Git integration settings
if os.environ.get('GIT_AUTHOR_NAME'):
    c.GitLabApp.git_author_name = os.environ.get('GIT_AUTHOR_NAME')
    c.GitLabApp.git_author_email = os.environ.get('GIT_AUTHOR_EMAIL')

# MLflow integration
if os.environ.get('MLFLOW_TRACKING_URI'):
    c.ServerApp.extra_env = {
        'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI')
    }

# =============================================================================
# Custom Extensions
# =============================================================================

# Load custom extensions
c.ServerApp.jpserver_extensions.update({
    'jupyter_resource_usage': True,
    'jupyter_ai': True,
    'jupyter_collaboration': True,
})

# =============================================================================
# Startup Scripts
# =============================================================================

# Custom startup script for environment setup
startup_script = """
import os
import sys
import warnings

# Suppress warnings in production
if os.environ.get('JUPYTER_ENV') == 'production':
    warnings.filterwarnings('ignore')

# Set up common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configure matplotlib for inline plotting
%matplotlib inline

# Set up pandas display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Jupyter environment initialized successfully!")
print(f"📊 Pandas {pd.__version__} | NumPy {np.__version__}")
print(f"🔍 Current working directory: {os.getcwd()}")
"""

# Write startup script
startup_dir = '/home/jovyan/.ipython/profile_default/startup'
if not os.path.exists(startup_dir):
    os.makedirs(startup_dir, exist_ok=True)

with open(os.path.join(startup_dir, '00-startup.py'), 'w') as f:
    f.write(startup_script) 