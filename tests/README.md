# Testing Framework

## Overview

The `tests` directory contains comprehensive testing suites for the Modern Data Stack Showcase platform. This testing framework ensures quality, reliability, and maintainability across all components of the data platform through multi-layered testing strategies.

## 🧪 **Testing Architecture**

### **Testing Philosophy**
- **Quality First**: Every component must have comprehensive test coverage
- **Fast Feedback**: Tests should run quickly in development cycles
- **Production Parity**: Tests should reflect real-world usage scenarios
- **Automated Validation**: All tests run automatically in CI/CD pipelines

## 📁 **Directory Structure**

```
tests/
├── README.md                    # This overview document
├── unit/                        # Unit tests for individual components
│   ├── test_ml_utils.py        # ML utilities testing
│   ├── test_data_processing.py # Data processing functions
│   ├── test_feature_engineering.py # Feature engineering validation
│   └── test_deployment.py      # Deployment utilities testing
├── integration/                 # Integration tests for component interactions
│   ├── test_ml_pipeline.py     # End-to-end ML pipeline testing
│   ├── test_data_flow.py       # Data flow between components
│   ├── test_api_endpoints.py   # API integration testing
│   └── test_monitoring.py      # Monitoring integration testing
└── e2e/                         # End-to-end tests for complete workflows
    ├── test_complete_ml_workflow.py # Full ML lifecycle testing
    ├── test_data_pipeline.py   # Complete data pipeline testing
    ├── test_deployment_workflow.py # Deployment workflow testing
    └── test_monitoring_alerts.py # Monitoring and alerting testing
```

## 🎯 **Testing Categories**

### **Unit Tests** (`unit/`)
Test individual functions and classes in isolation:

- **ML Utilities**: Test feature engineering, model training, evaluation functions
- **Data Processing**: Test data transformation and validation utilities
- **API Functions**: Test individual API endpoints and utilities
- **Configuration**: Test configuration management and validation

**Characteristics:**
- **Fast execution**: < 1 second per test
- **Isolated**: No external dependencies
- **Comprehensive**: 90%+ code coverage
- **Mocked dependencies**: Use mocks for external services

### **Integration Tests** (`integration/`)
Test interactions between multiple components:

- **ML Pipeline**: Test model training → evaluation → deployment flow
- **Data Flow**: Test data movement between dbt → ML → monitoring
- **API Integration**: Test API interactions with databases and services
- **Monitoring**: Test metrics collection and alerting systems

**Characteristics:**
- **Moderate execution**: 5-30 seconds per test
- **Component interaction**: Test real component interfaces
- **Database usage**: Use test databases with sample data
- **Service dependencies**: Test with real service interactions

### **End-to-End Tests** (`e2e/`)
Test complete user workflows and business scenarios:

- **Complete ML Workflow**: Feature engineering → training → deployment → monitoring
- **Data Pipeline**: Raw data → cleaned → analytics → business insights
- **Deployment Workflow**: Code → build → test → deploy → monitor
- **User Scenarios**: Realistic user interaction patterns

**Characteristics:**
- **Longer execution**: 1-10 minutes per test
- **Full system**: Test complete system integration
- **Production-like**: Use production-similar environments
- **User perspective**: Test from end-user viewpoint

## 🛠️ **Testing Tools & Frameworks**

### **Core Testing Stack**
- **pytest**: Primary testing framework with fixtures and parametrization
- **pytest-cov**: Code coverage measurement and reporting
- **pytest-mock**: Mocking and stubbing for unit tests
- **pytest-xdist**: Parallel test execution for faster feedback

### **Specialized Testing Tools**
- **Great Expectations**: Data quality and validation testing
- **MLflow**: Model performance and regression testing
- **Docker**: Containerized testing environments
- **Kubernetes**: Infrastructure testing and validation

### **Data & ML Testing**
- **pandas.testing**: DataFrame comparison and validation
- **scikit-learn.metrics**: Model performance validation
- **SHAP**: Model interpretability testing
- **Faker**: Synthetic test data generation

## 🚀 **Running Tests**

### **All Tests**
```bash
# Run complete test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel
pytest -n auto
```

### **By Category**
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only
pytest tests/e2e/
```

### **Specific Test Files**
```bash
# Run specific test file
pytest tests/unit/test_ml_utils.py

# Run specific test function
pytest tests/unit/test_ml_utils.py::test_feature_engineering

# Run with verbose output
pytest -v tests/unit/test_ml_utils.py
```

### **CI/CD Testing**
```bash
# Fast test suite for CI
pytest tests/unit/ tests/integration/ --maxfail=1 --tb=short

# Full test suite for release
pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml
```

## 📊 **Test Data Management**

### **Test Data Strategy**
- **Synthetic Data**: Generated using Faker and custom generators
- **Fixture Data**: Small, focused datasets for specific test scenarios
- **Sample Data**: Realistic data samples for integration testing
- **Mock Data**: Mocked external service responses

### **Data Generation**
```python
# Example test data generation
import pytest
from faker import Faker
import pandas as pd

@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing."""
    fake = Faker()
    return pd.DataFrame({
        'product_id': [fake.uuid4() for _ in range(100)],
        'sales_amount': [fake.pydecimal(left_digits=3, right_digits=2) for _ in range(100)],
        'date': [fake.date() for _ in range(100)]
    })
```

### **Test Environment Setup**
```python
@pytest.fixture(scope="session")
def test_database():
    """Set up test database for integration tests."""
    # Create test database
    # Insert test data
    # Yield database connection
    # Clean up after tests
```

## 📈 **Quality Metrics & Coverage**

### **Coverage Requirements**
- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: 80%+ API endpoint coverage
- **End-to-End Tests**: 100% critical user journey coverage

### **Performance Benchmarks**
- **Unit Tests**: Complete in < 2 minutes
- **Integration Tests**: Complete in < 10 minutes  
- **End-to-End Tests**: Complete in < 30 minutes

### **Quality Gates**
- **All tests pass**: 100% pass rate required
- **Coverage threshold**: Minimum coverage requirements met
- **Performance regression**: No significant performance degradation
- **Security validation**: Security tests pass

## 🔧 **Test Configuration**

### **pytest.ini Configuration**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=90
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    ml: Machine learning tests
    data: Data processing tests
```

### **Environment Variables**
```bash
# Test environment configuration
export TESTING=true
export DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
export MOCK_EXTERNAL_APIS=true
export LOG_LEVEL=DEBUG
```

## 🎯 **Testing Best Practices**

### **Test Writing Guidelines**
1. **Descriptive Names**: Test names should clearly describe what is being tested
2. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and validation
3. **Single Responsibility**: Each test should validate one specific behavior
4. **Independent Tests**: Tests should not depend on other tests

### **Example Test Structure**
```python
def test_feature_engineering_creates_polynomial_features():
    """Test that feature engineering creates polynomial features correctly."""
    # Arrange
    input_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    expected_columns = ['feature1', 'feature2', 'feature1^2', 'feature2^2', 'feature1*feature2']
    
    # Act
    result = create_polynomial_features(input_data, degree=2)
    
    # Assert
    assert list(result.columns) == expected_columns
    assert len(result) == len(input_data)
    assert result['feature1^2'].iloc[0] == 1  # 1^2 = 1
```

### **Mock Usage Guidelines**
```python
@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing."""
    with patch('mlflow.tracking.MlflowClient') as mock_client:
        mock_client.return_value.log_metric.return_value = None
        yield mock_client

def test_model_logging_with_mlflow(mock_mlflow_client):
    """Test that model metrics are logged to MLflow."""
    # Test implementation with mocked MLflow
```

## 🔍 **Test Debugging & Troubleshooting**

### **Common Issues**
- **Flaky Tests**: Tests that occasionally fail due to timing or race conditions
- **Test Data Conflicts**: Tests interfering with each other's data
- **Environment Dependencies**: Tests failing due to missing dependencies
- **Performance Issues**: Tests taking too long to execute

### **Debugging Techniques**
```bash
# Run single test with detailed output
pytest -vvv -s tests/unit/test_ml_utils.py::test_specific_function

# Run test with debugger
pytest --pdb tests/unit/test_ml_utils.py::test_specific_function

# Run failed tests only
pytest --lf

# Run tests with detailed tracing
pytest --trace
```

## 📚 **Documentation & Resources**

### **Test Documentation**
- **Test Plans**: Detailed test scenarios and expected outcomes
- **Coverage Reports**: HTML coverage reports in `htmlcov/` directory
- **Performance Reports**: Test execution time analysis
- **Quality Metrics**: Test quality and reliability metrics

### **Additional Resources**
- **pytest Documentation**: https://docs.pytest.org/
- **Testing Best Practices**: Internal wiki and guidelines
- **CI/CD Integration**: Documentation for automated testing
- **Performance Testing**: Load and stress testing procedures

---

**🎯 Testing ensures the reliability and quality of our Modern Data Stack platform. Every component is validated through comprehensive testing strategies.** 