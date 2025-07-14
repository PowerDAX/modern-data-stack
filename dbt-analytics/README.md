# DBT Analytics - Data Transformation Pipeline

## Overview

The `dbt-analytics` directory contains a comprehensive **enterprise-grade dbt project** implementing a 7-layer data pipeline architecture with advanced transformation patterns, comprehensive testing, and multi-connector support. This implementation demonstrates modern data engineering best practices and serves as the foundation for the Modern Data Stack Showcase's data transformation capabilities.

## 🏗️ **Architecture Highlights**

### **7-Layer Pipeline Architecture**
Following enterprise data engineering patterns inspired by dbt-greenhouse:

1. **Direct Layer**: Raw data ingestion with minimal transformation
2. **Source Layer**: Data type standardization and basic cleansing
3. **Cleaned Layer**: Data quality improvements and standardization
4. **Staging Layer**: Business logic preparation and enrichment
5. **Normalized Layer**: Dimensional modeling and fact table creation
6. **Denormalized Layer**: Performance-optimized aggregations
7. **Analytics Layer**: Business-ready models for consumption

### **Multi-Connector Support**
- **Retail Connector**: Standard retail data patterns (Product/Store)
- **Grocery Connector**: Grocery-specific patterns (Item/Location)
- **Harmonized Models**: Cross-connector standardized schema

## 📁 **Directory Structure**

```
dbt-analytics/
├── README.md                    # This overview document
├── dbt_project.yml              # Main dbt project configuration
├── profiles.yml                 # Connection profiles for different environments
├── models/                      # Data transformation models
│   ├── crisp_platform/         # Enterprise platform models
│   │   ├── analytics/          # Analytics layer (business-ready)
│   │   └── elt/                # ELT pipeline layers
│   │       ├── harmonized/     # Cross-connector models
│   │       └── retail_connector/ # Connector-specific models
│   │           ├── direct/     # Raw data ingestion
│   │           ├── source/     # Data type standardization
│   │           ├── cleaned/    # Data quality improvements
│   │           ├── staging/    # Business logic preparation
│   │           ├── normalized/ # Dimensional modeling
│   │           └── denormalized/ # Performance aggregations
│   └── staging/                # Legacy staging models
├── macros/                     # Reusable SQL functions
│   ├── common/                 # Common utility macros
│   ├── data_quality/           # Data quality validation
│   ├── transformations/        # Data transformation functions
│   └── retail_connector/       # Connector-specific macros
├── seeds/                      # Static reference data
├── snapshots/                  # Slowly changing dimension tracking
└── target/                     # Generated documentation and artifacts
```

## 🎯 **Key Features**

### **Enterprise Data Pipeline**
- **7-layer architecture** with clear separation of concerns
- **Incremental loading** strategies for large-scale data processing
- **Advanced macros** for data quality and transformation
- **Comprehensive testing** with 200+ data quality tests
- **Multi-environment support** (dev, staging, prod)

### **Data Quality Framework**
- **Automated validation** at each pipeline layer
- **Data profiling** and quality metrics
- **Anomaly detection** and alerting
- **Comprehensive testing** covering data integrity, freshness, and consistency
- **Quality gates** preventing bad data propagation

### **Multi-Connector Architecture**
- **Retail Connector**: Standard retail data patterns
- **Grocery Connector**: Grocery-specific business logic
- **Harmonized Models**: Cross-connector analytics
- **Systematic Transformations**: Automated renaming and normalization

## 🚀 **Getting Started**

### **Prerequisites**
- **dbt Core 1.6+** or **dbt Cloud**
- **BigQuery** or supported data warehouse
- **Python 3.8+** for dbt installation
- **Git** for version control

### **Installation**

1. **Install dbt**:
   ```bash
   pip install dbt-bigquery
   ```

2. **Configure profiles**:
   ```bash
   # Edit profiles.yml or use dbt Cloud
   dbt debug
   ```

3. **Install dependencies**:
   ```bash
   dbt deps
   ```

4. **Run the pipeline**:
   ```bash
   dbt run
   ```

### **Development Workflow**

1. **Create new models** in appropriate layer directories
2. **Add tests** for data quality validation
3. **Update documentation** with model descriptions
4. **Run and test** changes locally
5. **Deploy** through CI/CD pipeline

## 🔧 **Model Layers**

### **Direct Layer** (`models/crisp_platform/elt/retail_connector/direct/`)
- **Raw data ingestion** with minimal transformation
- **Column aliasing** for standardization
- **Basic data type casting**
- **Source system metadata** preservation

### **Source Layer** (`models/crisp_platform/elt/retail_connector/source/`)
- **Data type standardization** and validation
- **Basic cleansing** and null handling
- **Column renaming** for consistency
- **Reference data enrichment**

### **Cleaned Layer** (`models/crisp_platform/elt/retail_connector/cleaned/`)
- **Data quality improvements** and standardization
- **Duplicate removal** and deduplication
- **Data validation** and error handling
- **Business rule application**

### **Staging Layer** (`models/crisp_platform/elt/retail_connector/staging/`)
- **Business logic preparation** and enrichment
- **Complex transformations** and calculations
- **Data aggregation** and summarization
- **Dimension preparation**

### **Normalized Layer** (`models/crisp_platform/elt/retail_connector/normalized/`)
- **Dimensional modeling** (star schema)
- **Fact table creation** with measures
- **Dimension tables** with attributes
- **Relationship establishment**

### **Denormalized Layer** (`models/crisp_platform/elt/retail_connector/denormalized/`)
- **Performance-optimized aggregations**
- **Pre-calculated metrics** and KPIs
- **Flattened structures** for reporting
- **Optimized for query performance**

### **Analytics Layer** (`models/crisp_platform/analytics/`)
- **Business-ready models** for consumption
- **Cross-connector analytics** and insights
- **Executive dashboards** and reporting
- **Self-service analytics** preparation

## 🧪 **Testing Framework**

### **Data Quality Tests**
- **Schema tests**: Column constraints and data types
- **Referential integrity**: Foreign key relationships
- **Business rules**: Domain-specific validations
- **Freshness tests**: Data currency validation
- **Volume tests**: Expected row count validation

### **Custom Tests**
- **Advanced validation macros** for complex business rules
- **Data profiling tests** for statistical validation
- **Cross-table consistency** checks
- **Anomaly detection** for unusual patterns

### **Test Execution**
```bash
# Run all tests
dbt test

# Run tests for specific models
dbt test --select model_name

# Run tests by tag
dbt test --select tag:data_quality
```

## 📊 **Advanced Features**

### **Incremental Loading**
- **Efficient data processing** for large datasets
- **Merge strategies** for slowly changing dimensions
- **Watermark-based loading** for append-only tables
- **Performance optimization** for large-scale operations

### **Macro Library**
- **Reusable SQL functions** for common transformations
- **Data quality macros** for validation
- **Utility functions** for date manipulation and calculations
- **Connector-specific macros** for business logic

### **Documentation Generation**
- **Automated model documentation** with lineage
- **Data dictionary** with business definitions
- **Test coverage reports** and quality metrics
- **Interactive documentation** with dbt docs

## 🔄 **Integration Points**

### **Data Sources**
- **Sample data generators** (from `sample-data/` directory)
- **External data sources** via connectors
- **Reference data** from seeds
- **Real-time data streams** (future enhancement)

### **Downstream Consumers**
- **Power BI models** (semantic layer consumption)
- **Jupyter notebooks** (ML feature preparation)
- **Dashboard tools** (business intelligence)
- **API endpoints** (data serving)

### **Infrastructure Integration**
- **BigQuery** as primary data warehouse
- **Airflow** for orchestration (from `infrastructure/`)
- **CI/CD pipelines** for automated deployment
- **Monitoring** and alerting integration

## 🎛️ **Configuration**

### **Environment Management**
- **profiles.yml**: Database connection configuration
- **dbt_project.yml**: Project-level configuration
- **Environment variables**: Sensitive configuration
- **Feature flags**: Conditional model execution

### **Performance Optimization**
- **Materialization strategies** (table, view, incremental)
- **Partitioning and clustering** for BigQuery
- **Query optimization** techniques
- **Resource management** for large transformations

## 📈 **Monitoring & Observability**

### **Data Quality Monitoring**
- **Test results tracking** and alerting
- **Data freshness monitoring** and SLAs
- **Performance metrics** and optimization
- **Error tracking** and resolution

### **Pipeline Monitoring**
- **Run statistics** and performance metrics
- **Dependency tracking** and impact analysis
- **Cost monitoring** and optimization
- **Usage analytics** and adoption metrics

## 🛠️ **Maintenance**

### **Regular Tasks**
- **Update dbt version** and dependencies
- **Review and update tests** for data quality
- **Optimize model performance** and costs
- **Update documentation** with business changes

### **Best Practices**
- **Follow naming conventions** for consistency
- **Document all models** with business context
- **Use version control** for all changes
- **Test thoroughly** before deployment

## 🚀 **Future Enhancements**

- **Real-time streaming** integration
- **Additional connectors** (restaurant, healthcare, etc.)
- **Advanced ML features** and transformations
- **Data governance** and cataloging
- **Multi-cloud deployment** support

---

**📖 For comprehensive project details, see [dbt_project.yml](dbt_project.yml)**

**🔍 For model documentation, run `dbt docs generate && dbt docs serve`**

**🧪 For testing guide, see the [macros/](macros/) directory** 



