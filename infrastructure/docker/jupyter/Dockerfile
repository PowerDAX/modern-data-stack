# Advanced Jupyter Lab environment for data science and analytics
FROM jupyter/scipy-notebook:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

# Install Python packages for advanced analytics
RUN pip install --no-cache-dir \
    # Data manipulation and analysis
    pandas==2.1.4 \
    numpy==1.26.2 \
    polars==0.20.3 \
    dask[complete]==2023.12.1 \
    # Visualization
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.17.0 \
    altair==5.2.0 \
    bokeh==3.3.2 \
    # Machine learning
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    lightgbm==4.1.0 \
    optuna==3.5.0 \
    # Deep learning
    tensorflow==2.15.0 \
    torch==2.1.2 \
    # MLOps and experiment tracking
    mlflow==2.9.2 \
    wandb==0.16.1 \
    # Database connectivity
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.23 \
    # dbt integration
    dbt-core==1.8.7 \
    dbt-postgres==1.8.2 \
    # Data quality
    great-expectations==0.18.16 \
    pandera==0.17.2 \
    # Statistical analysis
    scipy==1.11.4 \
    statsmodels==0.14.1 \
    # Time series analysis
    prophet==1.1.5 \
    pmdarima==2.0.4 \
    # Text processing
    nltk==3.8.1 \
    spacy==3.7.2 \
    # API and web scraping
    requests==2.31.0 \
    beautifulsoup4==4.12.2 \
    # Jupyter extensions
    jupyterlab-git==0.50.0 \
    jupyterlab-lsp==5.0.1 \
    jupyter-dash==0.4.2 \
    # Development tools
    black==23.12.1 \
    isort==5.13.2 \
    pre-commit==3.6.0

# Install JupyterLab extensions
RUN jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/git \
    @jupyterlab/toc

# Install additional kernels
RUN python -m ipykernel install --user --name=showcase-analytics --display-name="Showcase Analytics"

# Create work directories
RUN mkdir -p /home/jovyan/work/{notebooks,data,dbt-analytics,models}

# Set working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["start-notebook.sh", "--LabApp.token=modern-data-stack-showcase"] 