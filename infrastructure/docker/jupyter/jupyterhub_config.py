# JupyterHub Configuration for Modern Data Stack Showcase
# Multi-user environment with specialized ML and DevOps containers

import os
from dockerspawner import DockerSpawner
from jupyter_client.localinterfaces import public_ips

# =============================================================================
# JupyterHub Configuration
# =============================================================================

# JupyterHub IP and port configuration
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8000

# Hub URL configuration
c.JupyterHub.hub_ip = '0.0.0.0'
c.JupyterHub.hub_port = 8081

# =============================================================================
# Database Configuration
# =============================================================================

# PostgreSQL database configuration
postgres_host = os.environ.get('POSTGRES_HOST', 'postgres')
postgres_db = os.environ.get('POSTGRES_DB', 'jupyterhub')
postgres_user = os.environ.get('POSTGRES_USER', 'jupyterhub')
postgres_password = os.environ.get('POSTGRES_PASSWORD', 'jupyterhub')

c.JupyterHub.db_url = f'postgresql://{postgres_user}:{postgres_password}@{postgres_host}:5432/{postgres_db}'

# =============================================================================
# Authentication Configuration
# =============================================================================

# Choose authentication method based on environment
auth_method = os.environ.get('JUPYTERHUB_AUTH_METHOD', 'native')

if auth_method == 'native':
    # Native authenticator with user registration
    c.JupyterHub.authenticator_class = 'nativeauthenticator.NativeAuthenticator'
    c.NativeAuthenticator.create_user_command = None
    c.NativeAuthenticator.open_signup = True
    c.NativeAuthenticator.ask_email_on_signup = True
    c.NativeAuthenticator.minimum_password_length = 8
    c.NativeAuthenticator.check_common_password = True
    c.NativeAuthenticator.allow_self_approval_for = {'admin'}
    
elif auth_method == 'oauth':
    # OAuth authenticator (GitHub, Google, etc.)
    c.JupyterHub.authenticator_class = 'oauthenticator.GitHubOAuthenticator'
    c.GitHubOAuthenticator.client_id = os.environ.get('GITHUB_CLIENT_ID')
    c.GitHubOAuthenticator.client_secret = os.environ.get('GITHUB_CLIENT_SECRET')
    c.GitHubOAuthenticator.oauth_callback_url = os.environ.get('OAUTH_CALLBACK_URL')
    
elif auth_method == 'ldap':
    # LDAP authenticator
    c.JupyterHub.authenticator_class = 'ldapauthenticator.LDAPAuthenticator'
    c.LDAPAuthenticator.server_address = os.environ.get('LDAP_SERVER')
    c.LDAPAuthenticator.bind_dn_template = os.environ.get('LDAP_BIND_DN_TEMPLATE')
    
else:
    # Dummy authenticator for development
    c.JupyterHub.authenticator_class = 'dummyauthenticator.DummyAuthenticator'
    c.DummyAuthenticator.password = os.environ.get('DUMMY_PASSWORD', 'password')

# =============================================================================
# Spawner Configuration
# =============================================================================

# Docker spawner configuration
c.JupyterHub.spawner_class = DockerSpawner

# Docker network configuration
c.DockerSpawner.network_name = os.environ.get('DOCKER_NETWORK_NAME', 'modern-stack-jupyter_default')
c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.remove = True

# Docker image selection based on user profile
def get_docker_image(spawner):
    """Select Docker image based on user profile or preference"""
    user_profile = spawner.user_options.get('profile', 'ml')
    
    if user_profile == 'ml':
        return 'modern-stack-jupyter-ml:latest'
    elif user_profile == 'devops':
        return 'modern-stack-jupyter-devops:latest'
    else:
        return 'modern-stack-jupyter-ml:latest'  # Default to ML

c.DockerSpawner.image = get_docker_image

# Volume mounts for persistent storage
c.DockerSpawner.volumes = {
    'jupyterhub-user-{username}': '/home/jovyan/work',
    'jupyterhub-shared': '/home/jovyan/shared',
    '/var/run/docker.sock': '/var/run/docker.sock',
    f'{os.getcwd()}/../../notebooks': '/home/jovyan/notebooks',
    f'{os.getcwd()}/../../dbt-analytics': '/home/jovyan/dbt-analytics',
    f'{os.getcwd()}/../../sample-data': '/home/jovyan/sample-data'
}

# Environment variables for spawned containers
c.DockerSpawner.environment = {
    'JUPYTERHUB_USER': '{username}',
    'JUPYTERHUB_SERVICE_PREFIX': '{service_prefix}',
    'JUPYTER_ENABLE_LAB': 'yes',
    'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
    'POSTGRES_HOST': postgres_host,
    'POSTGRES_DB': 'mlflow',
    'POSTGRES_USER': 'mlflow',
    'POSTGRES_PASSWORD': postgres_password
}

# Resource limits
c.DockerSpawner.mem_limit = '4G'
c.DockerSpawner.cpu_limit = 2.0

# Container lifecycle management
c.DockerSpawner.start_timeout = 300
c.DockerSpawner.http_timeout = 120

# =============================================================================
# User Profiles and Options
# =============================================================================

# Profile list for user selection
c.Spawner.profile_list = [
    {
        'display_name': 'Machine Learning Environment',
        'description': 'Python environment optimized for ML/AI workflows with TensorFlow, PyTorch, and MLflow',
        'default': True,
        'kubespawner_override': {
            'image': 'modern-stack-jupyter-ml:latest',
            'cpu_limit': 2.0,
            'mem_limit': '4G',
            'extra_resource_limits': {'nvidia.com/gpu': '1'}
        }
    },
    {
        'display_name': 'DevOps & Infrastructure',
        'description': 'Environment for DevOps automation with Docker, Kubernetes, and cloud tools',
        'kubespawner_override': {
            'image': 'modern-stack-jupyter-devops:latest',
            'cpu_limit': 1.0,
            'mem_limit': '2G',
            'privileged': True
        }
    },
    {
        'display_name': 'Data Engineering',
        'description': 'Environment for data engineering with dbt, SQL, and data processing tools',
        'kubespawner_override': {
            'image': 'modern-stack-jupyter-ml:latest',
            'cpu_limit': 1.5,
            'mem_limit': '3G'
        }
    }
]

# =============================================================================
# Administrative Configuration
# =============================================================================

# Admin users
c.Authenticator.admin_users = {'admin', 'dataeng', 'mlops'}

# Allowed users (if using whitelist)
c.Authenticator.allowed_users = {
    'admin', 'dataeng', 'mlops', 'analyst', 'scientist', 'engineer'
}

# =============================================================================
# Services Configuration
# =============================================================================

# Idle culler service
c.JupyterHub.services = [
    {
        'name': 'idle-culler',
        'command': [
            'python3', '-m', 'jupyterhub_idle_culler', 
            '--timeout=3600',  # 1 hour timeout
            '--cull-every=300',  # Check every 5 minutes
            '--concurrency=10',
            '--max-age=86400'  # Max age 24 hours
        ],
    },
    {
        'name': 'prometheus',
        'command': ['jupyterhub-prometheus'],
        'port': 9090,
        'environment': {
            'PROMETHEUS_METRICS_PORT': '9090'
        }
    }
]

# =============================================================================
# Security Configuration
# =============================================================================

# SSL configuration
ssl_enabled = os.environ.get('SSL_ENABLED', 'false').lower() == 'true'
if ssl_enabled:
    c.JupyterHub.ssl_cert = '/srv/jupyterhub/ssl/cert.pem'
    c.JupyterHub.ssl_key = '/srv/jupyterhub/ssl/key.pem'
    c.JupyterHub.port = 8443

# Cookie secret
c.JupyterHub.cookie_secret = bytes.fromhex(
    os.environ.get('JUPYTERHUB_CRYPT_KEY', 'a' * 64)
)

# Proxy configuration
c.ConfigurableHTTPProxy.auth_token = os.environ.get('JUPYTERHUB_PROXY_TOKEN')

# =============================================================================
# Logging Configuration
# =============================================================================

# Log level and format
c.JupyterHub.log_level = 'INFO'
c.JupyterHub.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Log file
c.JupyterHub.extra_log_file = '/var/log/jupyterhub/jupyterhub.log'

# =============================================================================
# UI Customization
# =============================================================================

# Custom logo and title
c.JupyterHub.logo_file = '/srv/jupyterhub/static/images/logo.png'
c.JupyterHub.template_paths = ['/srv/jupyterhub/templates']
c.JupyterHub.extra_static_paths = ['/srv/jupyterhub/static']

# Custom CSS and JavaScript
c.JupyterHub.template_vars = {
    'custom_css': '/static/css/custom.css',
    'custom_js': '/static/js/custom.js'
}

# =============================================================================
# Monitoring and Metrics
# =============================================================================

# Enable metrics collection
c.JupyterHub.statsd_host = os.environ.get('STATSD_HOST', 'localhost')
c.JupyterHub.statsd_port = int(os.environ.get('STATSD_PORT', '8125'))
c.JupyterHub.statsd_prefix = 'jupyterhub'

# =============================================================================
# Backup and Recovery
# =============================================================================

# Database backup configuration
c.JupyterHub.db_kwargs = {
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'echo': False
}

# =============================================================================
# Development vs Production Settings
# =============================================================================

if os.environ.get('JUPYTERHUB_ENV') == 'development':
    # Development-specific settings
    c.JupyterHub.debug = True
    c.LocalProcessSpawner.debug = True
    c.DockerSpawner.debug = True
    c.Spawner.debug = True
    
    # Allow all users in development
    c.Authenticator.allow_all = True
    
elif os.environ.get('JUPYTERHUB_ENV') == 'production':
    # Production-specific settings
    c.JupyterHub.debug = False
    c.JupyterHub.log_level = 'WARNING'
    
    # Strict user management in production
    c.Authenticator.allow_all = False
    c.Authenticator.allowed_users = c.Authenticator.allowed_users
    
    # Enhanced security
    c.JupyterHub.cookie_max_age_days = 1
    c.JupyterHub.reset_db = False

# =============================================================================
# Custom Hooks
# =============================================================================

def pre_spawn_hook(spawner):
    """Hook to run before spawning user container"""
    username = spawner.user.name
    print(f"Pre-spawn hook: Setting up environment for user {username}")
    
    # Set user-specific environment variables
    spawner.environment['JUPYTERHUB_USER'] = username
    spawner.environment['USER_HOME'] = f'/home/jovyan'
    
    # Log spawning activity
    print(f"Spawning container for user {username} with image {spawner.image}")

def post_stop_hook(spawner):
    """Hook to run after stopping user container"""
    username = spawner.user.name
    print(f"Post-stop hook: Cleaning up for user {username}")
    
    # Perform cleanup tasks
    # (backup user data, clean temporary files, etc.)

c.Spawner.pre_spawn_hook = pre_spawn_hook
c.Spawner.post_stop_hook = post_stop_hook

# =============================================================================
# Load Balancing and Scaling
# =============================================================================

# If using multiple JupyterHub instances
c.JupyterHub.hub_connect_ip = os.environ.get('JUPYTERHUB_HUB_CONNECT_IP')
c.JupyterHub.hub_connect_port = int(os.environ.get('JUPYTERHUB_HUB_CONNECT_PORT', '8081'))

# Cleanup settings
c.JupyterHub.cleanup_servers = True
c.JupyterHub.cleanup_proxy = True 