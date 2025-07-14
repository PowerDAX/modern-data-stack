# ML-Focused Jupyter Startup Script
# Sets up machine learning environment and common imports

import os
import sys
import warnings

# Suppress warnings in production
if os.environ.get('JUPYTER_ENV') == 'production':
    warnings.filterwarnings('ignore')

# Set up common ML imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ML libraries
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    print(f"✅ Scikit-learn {sklearn.__version__} loaded")
except ImportError:
    print("❌ Scikit-learn not available")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} loaded")
    # Configure TensorFlow for memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"📊 GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration warning: {e}")
except ImportError:
    print("❌ TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    print(f"✅ PyTorch {torch.__version__} loaded")
    if torch.cuda.is_available():
        print(f"📊 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("📊 CUDA not available, using CPU")
except ImportError:
    print("❌ PyTorch not available")

try:
    import mlflow
    print(f"✅ MLflow {mlflow.__version__} loaded")
    # Set up MLflow tracking URI
    if os.environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
        print(f"📊 MLflow tracking URI: {mlflow.get_tracking_uri()}")
except ImportError:
    print("❌ MLflow not available")

# Configure matplotlib for inline plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('seaborn-v0_8')

# Set up pandas display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Set up plotting style
sns.set_palette("husl")

# Set up numpy print options
np.set_printoptions(precision=4, suppress=True)

# Configure random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if 'sklearn' in sys.modules:
    from sklearn.utils import check_random_state
    check_random_state(RANDOM_SEED)
if 'tensorflow' in sys.modules:
    tf.random.set_seed(RANDOM_SEED)
if 'torch' in sys.modules:
    torch.manual_seed(RANDOM_SEED)

# Set up common ML utility functions
def load_sample_data(name='iris'):
    """Load common sample datasets for ML experimentation"""
    if name == 'iris':
        from sklearn.datasets import load_iris
        return load_iris(return_X_y=True, as_frame=True)
    elif name == 'boston':
        from sklearn.datasets import load_boston
        return load_boston(return_X_y=True)
    elif name == 'digits':
        from sklearn.datasets import load_digits
        return load_digits(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def setup_mlflow_experiment(experiment_name):
    """Set up MLflow experiment with proper configuration"""
    if 'mlflow' in sys.modules:
        mlflow.set_experiment(experiment_name)
        print(f"📊 MLflow experiment set to: {experiment_name}")
    else:
        print("❌ MLflow not available for experiment setup")

def quick_model_evaluation(model, X_test, y_test, task_type='classification'):
    """Quick evaluation of trained models"""
    if task_type == 'classification':
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        return accuracy
    elif task_type == 'regression':
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"📊 MSE: {mse:.4f}")
        print(f"📊 R²: {r2:.4f}")
        return mse, r2

# Set up environment variables for ML workflows
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Display environment information
print("🚀 ML Jupyter environment initialized successfully!")
print(f"📊 Pandas {pd.__version__} | NumPy {np.__version__}")
print(f"🔍 Current working directory: {os.getcwd()}")
print(f"🎲 Random seed set to: {RANDOM_SEED}")
print(f"🧠 Python version: {sys.version}")
print(f"💻 Available cores: {os.cpu_count()}")

# Check for GPU availability
if 'tensorflow' in sys.modules:
    print(f"🎮 TensorFlow GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}")
if 'torch' in sys.modules:
    print(f"🎮 PyTorch CUDA: {torch.cuda.is_available()}")

print("\n🛠️  Available utility functions:")
print("   - load_sample_data(name): Load sample datasets")
print("   - setup_mlflow_experiment(name): Set up MLflow experiment")
print("   - quick_model_evaluation(model, X_test, y_test): Quick model evaluation")
print("\n🎯 Ready for machine learning workflows!") 