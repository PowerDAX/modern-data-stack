# %% [markdown]
# # Model Deployment and Serving
# 
# This notebook provides comprehensive model deployment capabilities including:
# - Model packaging and versioning
# - REST API creation for model serving
# - Containerization with Docker
# - Production deployment strategies
# - Load testing and performance optimization
# - Monitoring and alerting setup
# 
# **Dependencies:** This notebook depends on models evaluated in `03-model-evaluation.py`

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

import pandas as pd
import numpy as np
import joblib
import json
import pickle
from datetime import datetime
import shutil
import tempfile
import requests
import time
from pathlib import Path
import yaml
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from ml_utils import ModelDeployment, ModelEvaluator
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## 2. Configuration and Model Loading

# %%
# Configuration
config = Config()

# MLflow configuration
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("retail-analytics-model-deployment")

# Load evaluation results to determine best model
try:
    evaluation_results = pd.read_csv(f"{config.output_path}/evaluation/model_comparison.csv")
    best_model_name = evaluation_results.iloc[0]['Model']
    print(f"Best model identified: {best_model_name}")
    
    # Load the best model
    model_path = f"{config.model_path}/{best_model_name.lower().replace(' ', '_')}_model.pkl"
    best_model = joblib.load(model_path)
    print(f"✓ Loaded best model: {best_model_name}")
    
    # Load feature names
    feature_names = pd.read_csv(f"{config.data_path}/feature_names.csv")['feature'].tolist()
    print(f"✓ Loaded {len(feature_names)} feature names")
    
except Exception as e:
    print(f"Error loading evaluation results: {e}")
    print("Please run 03-model-evaluation.py first")
    raise

# %% [markdown]
# ## 3. Model Packaging and Versioning

# %%
# Initialize deployment utility
deployment = ModelDeployment()

# Create deployment package
print("Creating deployment package...")

# Model metadata
model_metadata = {
    'model_name': best_model_name,
    'version': '1.0.0',
    'creation_date': datetime.now().isoformat(),
    'feature_names': feature_names,
    'model_type': str(type(best_model).__name__),
    'framework': 'scikit-learn',
    'requirements': [
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'joblib>=1.0.0'
    ]
}

# Package model with metadata
deployment_package = deployment.package_model(
    model=best_model,
    metadata=model_metadata,
    feature_names=feature_names
)

print(f"✓ Model packaged successfully")
print(f"  - Model: {best_model_name}")
print(f"  - Version: {model_metadata['version']}")
print(f"  - Features: {len(feature_names)}")
print(f"  - Framework: {model_metadata['framework']}")

# %% [markdown]
# ## 4. REST API Creation

# %%
# Create Flask application for model serving
def create_model_api():
    """Create Flask API for model serving"""
    
    app = Flask(__name__)
    CORS(app)
    
    # Global variables for the API
    model = None
    metadata = None
    feature_names_api = None
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model is not None
        })
    
    @app.route('/info', methods=['GET'])
    def model_info():
        """Model information endpoint"""
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        return jsonify({
            'model_name': metadata['model_name'],
            'version': metadata['version'],
            'model_type': metadata['model_type'],
            'framework': metadata['framework'],
            'feature_count': len(feature_names_api),
            'creation_date': metadata['creation_date']
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Prediction endpoint"""
        try:
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Get request data
            data = request.get_json()
            
            if 'features' not in data:
                return jsonify({'error': 'Missing features in request'}), 400
            
            # Convert to DataFrame
            features = pd.DataFrame([data['features']])
            
            # Validate features
            if len(features.columns) != len(feature_names_api):
                return jsonify({
                    'error': f'Expected {len(feature_names_api)} features, got {len(features.columns)}'
                }), 400
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get prediction probability if available
            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features)[0].tolist()
            
            # Return results
            result = {
                'prediction': int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
                'prediction_proba': prediction_proba,
                'model_name': metadata['model_name'],
                'version': metadata['version'],
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        """Batch prediction endpoint"""
        try:
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Get request data
            data = request.get_json()
            
            if 'features' not in data:
                return jsonify({'error': 'Missing features in request'}), 400
            
            # Convert to DataFrame
            features = pd.DataFrame(data['features'])
            
            # Validate features
            if len(features.columns) != len(feature_names_api):
                return jsonify({
                    'error': f'Expected {len(feature_names_api)} features, got {len(features.columns)}'
                }), 400
            
            # Make predictions
            predictions = model.predict(features)
            
            # Get prediction probabilities if available
            predictions_proba = None
            if hasattr(model, 'predict_proba'):
                predictions_proba = model.predict_proba(features).tolist()
            
            # Return results
            result = {
                'predictions': [int(p) if isinstance(p, (np.integer, np.int64)) else float(p) for p in predictions],
                'predictions_proba': predictions_proba,
                'count': len(predictions),
                'model_name': metadata['model_name'],
                'version': metadata['version'],
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Load model into API
    def load_model_into_api():
        nonlocal model, metadata, feature_names_api
        model = best_model
        metadata = model_metadata
        feature_names_api = feature_names
        logger.info(f"Model loaded into API: {metadata['model_name']}")
    
    # Load the model
    load_model_into_api()
    
    return app

# Create the API
model_api = create_model_api()

print("✓ REST API created successfully")
print("  Available endpoints:")
print("    - GET  /health - Health check")
print("    - GET  /info - Model information")
print("    - POST /predict - Single prediction")
print("    - POST /predict/batch - Batch predictions")

# %% [markdown]
# ## 5. API Testing and Validation

# %%
# Test the API endpoints
def test_api_endpoints():
    """Test API endpoints with sample data"""
    
    # Start the Flask app in a separate thread
    def run_api():
        model_api.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Wait for API to start
    time.sleep(2)
    
    base_url = "http://127.0.0.1:5001"
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        
        # Test info endpoint
        print("\nTesting info endpoint...")
        response = requests.get(f"{base_url}/info")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        
        # Test single prediction
        print("\nTesting single prediction...")
        # Create sample features (using zeros for simplicity)
        sample_features = {f"feature_{i}": 0.0 for i in range(len(feature_names))}
        
        response = requests.post(
            f"{base_url}/predict",
            json={'features': sample_features}
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        
        # Test batch prediction
        print("\nTesting batch prediction...")
        batch_features = [sample_features, sample_features]
        
        response = requests.post(
            f"{base_url}/predict/batch",
            json={'features': batch_features}
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response keys: {list(response.json().keys())}")
        
        print("\n✓ API testing completed successfully")
        
    except Exception as e:
        print(f"API testing error: {e}")
    
    return True

# Run API tests
test_api_endpoints()

# %% [markdown]
# ## 6. Docker Containerization

# %%
# Create Docker configuration
docker_config = {
    'dockerfile_content': f'''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]
''',
    'requirements_txt': '''
Flask==2.3.3
Flask-CORS==4.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.0.0
mlflow>=2.0.0
requests>=2.28.0
''',
    'app_py': '''
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata
model = joblib.load('model.pkl')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
feature_names = metadata['feature_names']

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/info', methods=['GET'])
def model_info():
    return jsonify({
        'model_name': metadata['model_name'],
        'version': metadata['version'],
        'model_type': metadata['model_type'],
        'framework': metadata['framework'],
        'feature_count': len(feature_names),
        'creation_date': metadata['creation_date']
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = pd.DataFrame([data['features']])
        
        if len(features.columns) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features.columns)}'
            }), 400
        
        prediction = model.predict(features)[0]
        
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features)[0].tolist()
        
        result = {
            'prediction': int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
            'prediction_proba': prediction_proba,
            'model_name': metadata['model_name'],
            'version': metadata['version'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = pd.DataFrame(data['features'])
        
        if len(features.columns) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features.columns)}'
            }), 400
        
        predictions = model.predict(features)
        
        predictions_proba = None
        if hasattr(model, 'predict_proba'):
            predictions_proba = model.predict_proba(features).tolist()
        
        result = {
            'predictions': [int(p) if isinstance(p, (np.integer, np.int64)) else float(p) for p in predictions],
            'predictions_proba': predictions_proba,
            'count': len(predictions),
            'model_name': metadata['model_name'],
            'version': metadata['version'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
}

# Create deployment directory
deployment_dir = f"{config.output_path}/deployment"
os.makedirs(deployment_dir, exist_ok=True)

# Write Docker files
with open(f"{deployment_dir}/Dockerfile", 'w') as f:
    f.write(docker_config['dockerfile_content'])

with open(f"{deployment_dir}/requirements.txt", 'w') as f:
    f.write(docker_config['requirements_txt'])

with open(f"{deployment_dir}/app.py", 'w') as f:
    f.write(docker_config['app_py'])

# Copy model and metadata
joblib.dump(best_model, f"{deployment_dir}/model.pkl")
with open(f"{deployment_dir}/metadata.json", 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✓ Docker configuration created")
print(f"  Deployment directory: {deployment_dir}")
print("  Files created:")
print("    - Dockerfile")
print("    - requirements.txt")
print("    - app.py")
print("    - model.pkl")
print("    - metadata.json")

# %% [markdown]
# ## 7. Kubernetes Deployment Configuration

# %%
# Create Kubernetes deployment configuration
k8s_config = {
    'deployment.yaml': f'''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {best_model_name.lower().replace(' ', '-')}-model
  labels:
    app: {best_model_name.lower().replace(' ', '-')}-model
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {best_model_name.lower().replace(' ', '-')}-model
  template:
    metadata:
      labels:
        app: {best_model_name.lower().replace(' ', '-')}-model
        version: v1
    spec:
      containers:
      - name: model-server
        image: {best_model_name.lower().replace(' ', '-')}-model:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_NAME
          value: "{best_model_name}"
        - name: MODEL_VERSION
          value: "1.0.0"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
''',
    'service.yaml': f'''
apiVersion: v1
kind: Service
metadata:
  name: {best_model_name.lower().replace(' ', '-')}-model-service
  labels:
    app: {best_model_name.lower().replace(' ', '-')}-model
spec:
  selector:
    app: {best_model_name.lower().replace(' ', '-')}-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
''',
    'ingress.yaml': f'''
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {best_model_name.lower().replace(' ', '-')}-model-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: {best_model_name.lower().replace(' ', '-')}-model.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {best_model_name.lower().replace(' ', '-')}-model-service
            port:
              number: 80
''',
    'hpa.yaml': f'''
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {best_model_name.lower().replace(' ', '-')}-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {best_model_name.lower().replace(' ', '-')}-model
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
}

# Create Kubernetes directory
k8s_dir = f"{deployment_dir}/k8s"
os.makedirs(k8s_dir, exist_ok=True)

# Write Kubernetes files
for filename, content in k8s_config.items():
    with open(f"{k8s_dir}/{filename}", 'w') as f:
        f.write(content)

print("✓ Kubernetes configuration created")
print(f"  Kubernetes directory: {k8s_dir}")
print("  Files created:")
print("    - deployment.yaml")
print("    - service.yaml")
print("    - ingress.yaml")
print("    - hpa.yaml")

# %% [markdown]
# ## 8. Load Testing and Performance Analysis

# %%
# Create load testing script
load_test_script = '''
import requests
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import statistics

def single_prediction_request(base_url, features):
    """Make a single prediction request"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json={'features': features},
            timeout=10
        )
        end_time = time.time()
        
        return {
            'status_code': response.status_code,
            'response_time': end_time - start_time,
            'success': response.status_code == 200
        }
    except Exception as e:
        return {
            'status_code': 0,
            'response_time': 0,
            'success': False,
            'error': str(e)
        }

def run_load_test(base_url, num_requests=100, num_threads=10):
    """Run load test against the API"""
    
    print(f"Starting load test:")
    print(f"  Base URL: {base_url}")
    print(f"  Requests: {num_requests}")
    print(f"  Threads: {num_threads}")
    print(f"  Started: {datetime.now()}")
    
    # Sample features for testing
    features = {f"feature_{i}": np.random.random() for i in range(20)}
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(single_prediction_request, base_url, features)
            for _ in range(num_requests)
        ]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    
    # Analyze results
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    
    response_times = [r['response_time'] for r in successful_requests]
    
    print(f"\\nLoad test completed:")
    print(f"  Total time: {end_time - start_time:.2f}s")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {len(successful_requests)}")
    print(f"  Failed: {len(failed_requests)}")
    print(f"  Success rate: {len(successful_requests)/len(results)*100:.1f}%")
    
    if response_times:
        print(f"  Average response time: {statistics.mean(response_times):.3f}s")
        print(f"  Median response time: {statistics.median(response_times):.3f}s")
        print(f"  95th percentile: {np.percentile(response_times, 95):.3f}s")
        print(f"  99th percentile: {np.percentile(response_times, 99):.3f}s")
        print(f"  Max response time: {max(response_times):.3f}s")
        print(f"  Min response time: {min(response_times):.3f}s")
    
    return {
        'total_requests': len(results),
        'successful_requests': len(successful_requests),
        'failed_requests': len(failed_requests),
        'success_rate': len(successful_requests)/len(results)*100,
        'response_times': response_times,
        'total_time': end_time - start_time
    }

if __name__ == "__main__":
    # Run load test
    base_url = "http://127.0.0.1:5001"
    results = run_load_test(base_url, num_requests=50, num_threads=5)
'''

# Write load test script
with open(f"{deployment_dir}/load_test.py", 'w') as f:
    f.write(load_test_script)

print("✓ Load testing script created")
print(f"  Script: {deployment_dir}/load_test.py")

# %% [markdown]
# ## 9. Monitoring and Alerting Configuration

# %%
# Create monitoring configuration
monitoring_config = {
    'prometheus.yml': f'''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'model-api'
    static_configs:
      - targets: ['{best_model_name.lower().replace(' ', '-')}-model-service:80']
    metrics_path: '/metrics'
    scrape_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
''',
    'alert_rules.yml': f'''
groups:
- name: model_alerts
  rules:
  - alert: ModelAPIDown
    expr: up{{job="{best_model_name.lower().replace(' ', '-')}-model"}} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Model API is down"
      description: "Model API has been down for more than 1 minute"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is above 1 second"
  
  - alert: HighErrorRate
    expr: rate(http_requests_total{{status=~"5.."|{job="{best_model_name.lower().replace(' ', '-')}-model"}}}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 10%"
  
  - alert: ModelPredictionAccuracyDrop
    expr: model_prediction_accuracy < 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model prediction accuracy dropped"
      description: "Model accuracy has dropped below 80%"
''',
    'grafana_dashboard.json': json.dumps({
        "dashboard": {
            "title": f"{best_model_name} Model Dashboard",
            "panels": [
                {
                    "title": "Request Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": f"rate(http_requests_total{{job=\"{best_model_name.lower().replace(' ', '-')}-model\"}}[5m])",
                            "legendFormat": "Requests/sec"
                        }
                    ]
                },
                {
                    "title": "Response Time",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": f"histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job=\"{best_model_name.lower().replace(' ', '-')}-model\"}}[5m]))",
                            "legendFormat": "95th percentile"
                        }
                    ]
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": f"rate(http_requests_total{{status=~\"5..\",job=\"{best_model_name.lower().replace(' ', '-')}-model\"}}[5m])",
                            "legendFormat": "Error rate"
                        }
                    ]
                }
            ]
        }
    }, indent=2)
}

# Create monitoring directory
monitoring_dir = f"{deployment_dir}/monitoring"
os.makedirs(monitoring_dir, exist_ok=True)

# Write monitoring files
for filename, content in monitoring_config.items():
    with open(f"{monitoring_dir}/{filename}", 'w') as f:
        f.write(content)

print("✓ Monitoring configuration created")
print(f"  Monitoring directory: {monitoring_dir}")
print("  Files created:")
print("    - prometheus.yml")
print("    - alert_rules.yml")
print("    - grafana_dashboard.json")

# %% [markdown]
# ## 10. Deployment Scripts and Automation

# %%
# Create deployment scripts
deployment_scripts = {
    'build_and_deploy.sh': f'''#!/bin/bash
set -e

MODEL_NAME="{best_model_name.lower().replace(' ', '-')}-model"
VERSION="1.0.0"
REGISTRY="your-registry.com"

echo "Building and deploying $MODEL_NAME..."

# Build Docker image
echo "Building Docker image..."
docker build -t $MODEL_NAME:$VERSION .
docker tag $MODEL_NAME:$VERSION $REGISTRY/$MODEL_NAME:$VERSION

# Push to registry
echo "Pushing to registry..."
docker push $REGISTRY/$MODEL_NAME:$VERSION

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$MODEL_NAME

# Run health check
echo "Running health check..."
kubectl port-forward service/$MODEL_NAME-service 8080:80 &
sleep 10
curl -f http://localhost:8080/health
kill %1

echo "Deployment completed successfully!"
''',
    'rollback.sh': f'''#!/bin/bash
set -e

MODEL_NAME="{best_model_name.lower().replace(' ', '-')}-model"
PREVIOUS_VERSION="$1"

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: $0 <previous_version>"
    exit 1
fi

echo "Rolling back $MODEL_NAME to version $PREVIOUS_VERSION..."

# Update deployment image
kubectl set image deployment/$MODEL_NAME model-server=your-registry.com/$MODEL_NAME:$PREVIOUS_VERSION

# Wait for rollback to complete
kubectl rollout status deployment/$MODEL_NAME

echo "Rollback completed successfully!"
''',
    'scale.sh': f'''#!/bin/bash
set -e

MODEL_NAME="{best_model_name.lower().replace(' ', '-')}-model"
REPLICAS="$1"

if [ -z "$REPLICAS" ]; then
    echo "Usage: $0 <number_of_replicas>"
    exit 1
fi

echo "Scaling $MODEL_NAME to $REPLICAS replicas..."

kubectl scale deployment/$MODEL_NAME --replicas=$REPLICAS

echo "Scaling completed successfully!"
''',
    'deploy.py': f'''#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import requests
from datetime import datetime

def run_command(cmd, cwd=None):
    """Run shell command"""
    print(f"Running: {{cmd}}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {{result.stderr}}")
        sys.exit(1)
    return result.stdout

def health_check(url, max_retries=10, delay=5):
    """Check if service is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{{url}}/health", timeout=10)
            if response.status_code == 200:
                print(f"✓ Health check passed")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"Health check failed, retrying in {{delay}}s...")
            time.sleep(delay)
    
    return False

def main():
    model_name = "{best_model_name.lower().replace(' ', '-')}-model"
    version = "1.0.0"
    
    print(f"Starting deployment of {{model_name}} v{{version}}")
    print(f"Timestamp: {{datetime.now()}}")
    
    # Build Docker image
    print("\\n1. Building Docker image...")
    run_command(f"docker build -t {{model_name}}:{{version}} .")
    
    # Deploy to Kubernetes
    print("\\n2. Deploying to Kubernetes...")
    run_command("kubectl apply -f k8s/")
    
    # Wait for deployment
    print("\\n3. Waiting for deployment to be ready...")
    run_command(f"kubectl wait --for=condition=available --timeout=300s deployment/{{model_name}}")
    
    # Health check
    print("\\n4. Running health check...")
    # This would need to be adapted based on your actual service URL
    service_url = "http://localhost:8080"  # Adjust as needed
    
    if health_check(service_url):
        print("\\n✅ Deployment completed successfully!")
    else:
        print("\\n❌ Deployment failed - health check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
}

# Create scripts directory
scripts_dir = f"{deployment_dir}/scripts"
os.makedirs(scripts_dir, exist_ok=True)

# Write deployment scripts
for filename, content in deployment_scripts.items():
    script_path = f"{scripts_dir}/{filename}"
    with open(script_path, 'w') as f:
        f.write(content)
    
    # Make shell scripts executable
    if filename.endswith('.sh'):
        os.chmod(script_path, 0o755)

print("✓ Deployment scripts created")
print(f"  Scripts directory: {scripts_dir}")
print("  Files created:")
print("    - build_and_deploy.sh")
print("    - rollback.sh")
print("    - scale.sh")
print("    - deploy.py")

# %% [markdown]
# ## 11. MLflow Model Registry Integration

# %%
# Register model with MLflow
try:
    print("Registering model with MLflow...")
    
    with mlflow.start_run(run_name="model_deployment_registration"):
        # Log model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=f"{best_model_name}_production"
        )
        
        # Log metadata
        mlflow.log_params(model_metadata)
        
        # Log deployment artifacts
        mlflow.log_artifacts(deployment_dir, "deployment")
        
        print(f"✓ Model registered: {best_model_name}_production")
        print(f"✓ Deployment artifacts logged")
        
except Exception as e:
    print(f"MLflow registration error: {e}")

# %% [markdown]
# ## 12. Performance Benchmarking

# %%
# Create performance benchmarking script
def benchmark_model_performance():
    """Benchmark model performance"""
    
    print("Running performance benchmarks...")
    
    # Load test data
    try:
        X_test = pd.read_csv(f"{config.data_path}/X_test_processed.csv")
        y_test = pd.read_csv(f"{config.data_path}/y_test.csv").squeeze()
        
        # Benchmark prediction speed
        print("\n1. Prediction Speed Benchmark:")
        
        # Single prediction benchmark
        start_time = time.time()
        for i in range(100):
            _ = best_model.predict(X_test.iloc[[i % len(X_test)]])
        single_pred_time = (time.time() - start_time) / 100
        
        print(f"   Single prediction: {single_pred_time*1000:.2f}ms")
        
        # Batch prediction benchmark
        batch_sizes = [1, 10, 100, 1000]
        for batch_size in batch_sizes:
            if batch_size <= len(X_test):
                start_time = time.time()
                _ = best_model.predict(X_test.iloc[:batch_size])
                batch_time = time.time() - start_time
                
                print(f"   Batch size {batch_size}: {batch_time*1000:.2f}ms ({batch_time/batch_size*1000:.2f}ms per prediction)")
        
        # Memory usage estimation
        print("\n2. Memory Usage Estimation:")
        import sys
        model_size = sys.getsizeof(best_model) / 1024 / 1024
        print(f"   Model size: {model_size:.2f} MB")
        
        # Throughput estimation
        print("\n3. Throughput Estimation:")
        predictions_per_second = 1 / single_pred_time
        print(f"   Predictions per second: {predictions_per_second:.0f}")
        print(f"   Predictions per minute: {predictions_per_second * 60:.0f}")
        print(f"   Predictions per hour: {predictions_per_second * 3600:.0f}")
        
        return {
            'single_prediction_time_ms': single_pred_time * 1000,
            'predictions_per_second': predictions_per_second,
            'model_size_mb': model_size
        }
        
    except Exception as e:
        print(f"Benchmarking error: {e}")
        return None

# Run benchmarks
benchmark_results = benchmark_model_performance()

# %% [markdown]
# ## 13. Deployment Summary and Documentation

# %%
# Create comprehensive deployment documentation
deployment_docs = f'''
# {best_model_name} Model Deployment Guide

## Overview
This deployment package contains everything needed to deploy the {best_model_name} model to production.

## Model Information
- **Model Name**: {best_model_name}
- **Version**: {model_metadata['version']}
- **Framework**: {model_metadata['framework']}
- **Features**: {len(feature_names)}
- **Model Type**: {model_metadata['model_type']}

## Deployment Architecture

### Components
1. **Model Server**: Flask-based REST API
2. **Containerization**: Docker container
3. **Orchestration**: Kubernetes deployment
4. **Monitoring**: Prometheus + Grafana
5. **Load Balancing**: Kubernetes service
6. **Auto-scaling**: Horizontal Pod Autoscaler

### API Endpoints
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py

# Test the API
curl -X GET http://localhost:5000/health
```

### Docker Deployment
```bash
# Build the image
docker build -t {best_model_name.lower().replace(' ', '-')}-model:1.0.0 .

# Run the container
docker run -p 5000:5000 {best_model_name.lower().replace(' ', '-')}-model:1.0.0

# Test the deployment
curl -X GET http://localhost:5000/health
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app={best_model_name.lower().replace(' ', '-')}-model

# Test the service
kubectl port-forward service/{best_model_name.lower().replace(' ', '-')}-model-service 8080:80
curl -X GET http://localhost:8080/health
```

## Performance Characteristics

### Benchmarks
- **Single Prediction**: {benchmark_results['single_prediction_time_ms']:.2f}ms
- **Throughput**: {benchmark_results['predictions_per_second']:.0f} predictions/second
- **Model Size**: {benchmark_results['model_size_mb']:.2f} MB

### Scaling
- **Default Replicas**: 3
- **Auto-scaling**: 3-10 replicas based on CPU/memory usage
- **Load Balancing**: Round-robin via Kubernetes service

## Monitoring and Alerting

### Metrics
- Request rate and response time
- Error rate and success rate
- Resource utilization (CPU, memory)
- Model prediction accuracy

### Alerts
- Model API down
- High response time (>1s)
- High error rate (>10%)
- Model accuracy drop (<80%)

## Maintenance

### Updates
```bash
# Deploy new version
./scripts/build_and_deploy.sh

# Rollback if needed
./scripts/rollback.sh <previous_version>
```

### Scaling
```bash
# Scale manually
./scripts/scale.sh 5

# Auto-scaling is configured via HPA
```

### Health Checks
```bash
# Check deployment health
kubectl get pods -l app={best_model_name.lower().replace(' ', '-')}-model

# Check service status
kubectl get svc {best_model_name.lower().replace(' ', '-')}-model-service

# View logs
kubectl logs -l app={best_model_name.lower().replace(' ', '-')}-model
```

## Security Considerations

1. **API Security**: Implement authentication and authorization
2. **Network Security**: Use HTTPS and secure ingress
3. **Container Security**: Regular security scans
4. **Data Privacy**: Ensure data encryption in transit and at rest

## Troubleshooting

### Common Issues
1. **Pod not starting**: Check resource limits and image availability
2. **Health check failing**: Verify application is listening on correct port
3. **High response time**: Check resource allocation and scaling
4. **Prediction errors**: Validate input data format and feature names

### Logs and Debugging
```bash
# View application logs
kubectl logs -l app={best_model_name.lower().replace(' ', '-')}-model --tail=100

# Describe pod for detailed status
kubectl describe pod <pod-name>

# Check resource usage
kubectl top pods -l app={best_model_name.lower().replace(' ', '-')}-model
```

## Support
For issues and questions, please check the monitoring dashboards and logs first.
'''

# Write deployment documentation
with open(f"{deployment_dir}/README.md", 'w') as f:
    f.write(deployment_docs)

print("✓ Deployment documentation created")
print(f"  Documentation: {deployment_dir}/README.md")

# %% [markdown]
# ## 14. Deployment Validation and Testing

# %%
# Create deployment validation checklist
validation_checklist = {
    'pre_deployment': [
        'Model artifacts are available and valid',
        'Docker image builds successfully',
        'Kubernetes manifests are valid',
        'Monitoring configuration is complete',
        'Load testing scripts are ready'
    ],
    'post_deployment': [
        'All pods are running and healthy',
        'Health check endpoint responds correctly',
        'Prediction endpoints work as expected',
        'Monitoring metrics are being collected',
        'Alerts are configured and functional',
        'Load balancer is distributing traffic',
        'Auto-scaling is working correctly'
    ],
    'performance_validation': [
        'Response time is within acceptable limits',
        'Throughput meets requirements',
        'Error rate is below threshold',
        'Resource utilization is optimal',
        'Prediction accuracy is maintained'
    ]
}

print("Deployment Validation Checklist:")
print("=" * 35)

for category, items in validation_checklist.items():
    print(f"\n{category.replace('_', ' ').title()}:")
    for item in items:
        print(f"  ☐ {item}")

# Save validation checklist
with open(f"{deployment_dir}/validation_checklist.md", 'w') as f:
    f.write("# Deployment Validation Checklist\n\n")
    for category, items in validation_checklist.items():
        f.write(f"## {category.replace('_', ' ').title()}\n\n")
        for item in items:
            f.write(f"- [ ] {item}\n")
        f.write("\n")

print(f"\n✓ Validation checklist saved: {deployment_dir}/validation_checklist.md")

# %% [markdown]
# ## 15. Deployment Complete

# %%
# Final deployment summary
print("🚀 MODEL DEPLOYMENT PACKAGE COMPLETE")
print("=" * 40)

print(f"\n📦 DEPLOYMENT PACKAGE CONTENTS:")
print(f"  Location: {deployment_dir}")
print(f"  Size: {len(os.listdir(deployment_dir))} files/directories")

print(f"\n🏆 MODEL DETAILS:")
print(f"  Model: {best_model_name}")
print(f"  Version: {model_metadata['version']}")
print(f"  Framework: {model_metadata['framework']}")
print(f"  Features: {len(feature_names)}")

print(f"\n🔧 DEPLOYMENT COMPONENTS:")
print(f"  ✅ Docker configuration")
print(f"  ✅ Kubernetes manifests")
print(f"  ✅ Monitoring setup")
print(f"  ✅ Load testing scripts")
print(f"  ✅ Deployment automation")
print(f"  ✅ Documentation")

print(f"\n📊 PERFORMANCE METRICS:")
if benchmark_results:
    print(f"  Latency: {benchmark_results['single_prediction_time_ms']:.2f}ms")
    print(f"  Throughput: {benchmark_results['predictions_per_second']:.0f} req/s")
    print(f"  Model Size: {benchmark_results['model_size_mb']:.2f} MB")

print(f"\n🚀 NEXT STEPS:")
print(f"  1. Review deployment documentation")
print(f"  2. Customize configuration for your environment")
print(f"  3. Run deployment validation")
print(f"  4. Execute deployment scripts")
print(f"  5. Monitor and maintain the deployment")

print(f"\n📁 KEY FILES:")
print(f"  📄 {deployment_dir}/README.md - Deployment guide")
print(f"  🐳 {deployment_dir}/Dockerfile - Container configuration")
print(f"  ⚙️ {deployment_dir}/k8s/ - Kubernetes manifests")
print(f"  📊 {deployment_dir}/monitoring/ - Monitoring setup")
print(f"  🔧 {deployment_dir}/scripts/ - Deployment automation")

print(f"\n✅ Ready for production deployment!")
print("   Proceed to 05-model-monitoring.py for monitoring setup")

# %%
# Log deployment completion to MLflow
with mlflow.start_run(run_name="deployment_completion"):
    mlflow.log_params({
        'model_name': best_model_name,
        'version': model_metadata['version'],
        'deployment_date': datetime.now().isoformat(),
        'components_created': len(os.listdir(deployment_dir))
    })
    
    if benchmark_results:
        mlflow.log_metrics({
            'single_prediction_time_ms': benchmark_results['single_prediction_time_ms'],
            'predictions_per_second': benchmark_results['predictions_per_second'],
            'model_size_mb': benchmark_results['model_size_mb']
        })

print("📈 Deployment metrics logged to MLflow") 