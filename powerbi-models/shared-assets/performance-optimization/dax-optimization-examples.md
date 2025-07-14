# Power BI Performance Optimization Examples

## Table of Contents
1. [DAX Optimization Techniques](#dax-optimization-techniques)
2. [Model Size Reduction Strategies](#model-size-reduction-strategies)
3. [Relationship Optimization](#relationship-optimization)
4. [Query Performance Optimization](#query-performance-optimization)
5. [Memory Management](#memory-management)
6. [Performance Testing Methodologies](#performance-testing-methodologies)
7. [Monitoring and Diagnostics](#monitoring-and-diagnostics)
8. [Best Practices Summary](#best-practices-summary)

---

## DAX Optimization Techniques

### 1. Use Variables for Complex Calculations

❌ **Poor Performance - Repeated Calculations:**
```dax
Sales Growth Rate = 
DIVIDE(
    SUM('Fact Sales'[sales_amount]) - 
    CALCULATE(
        SUM('Fact Sales'[sales_amount]),
        SAMEPERIODLASTYEAR('Dim Calendar'[Date])
    ),
    CALCULATE(
        SUM('Fact Sales'[sales_amount]),
        SAMEPERIODLASTYEAR('Dim Calendar'[Date])
    )
)
```

✅ **Optimized - Using Variables:**
```dax
Sales Growth Rate = 
VAR CurrentSales = SUM('Fact Sales'[sales_amount])
VAR PreviousYearSales = 
    CALCULATE(
        SUM('Fact Sales'[sales_amount]),
        SAMEPERIODLASTYEAR('Dim Calendar'[Date])
    )
RETURN
    DIVIDE(CurrentSales - PreviousYearSales, PreviousYearSales)
```

### 2. Avoid Expensive Iterator Functions

❌ **Poor Performance - SUMX with Complex Logic:**
```dax
Complex Calculation = 
SUMX(
    'Fact Sales',
    IF(
        'Fact Sales'[sales_amount] > 1000,
        'Fact Sales'[sales_amount] * 1.1,
        'Fact Sales'[sales_amount] * 0.9
    )
)
```

✅ **Optimized - Pre-calculated Column:**
```dax
// Create calculated column in table
Adjusted Sales Amount = 
IF(
    'Fact Sales'[sales_amount] > 1000,
    'Fact Sales'[sales_amount] * 1.1,
    'Fact Sales'[sales_amount] * 0.9
)

// Use simple SUM in measure
Complex Calculation = SUM('Fact Sales'[Adjusted Sales Amount])
```

### 3. Use CALCULATE Efficiently

❌ **Poor Performance - Multiple CALCULATE Calls:**
```dax
Multi Filter Measure = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    FILTER(
        'Dim Product',
        'Dim Product'[Product Category] = "Electronics"
    )
) +
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    FILTER(
        'Dim Product',
        'Dim Product'[Product Category] = "Clothing"
    )
)
```

✅ **Optimized - Single CALCULATE with OR:**
```dax
Multi Filter Measure = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    'Dim Product'[Product Category] IN {"Electronics", "Clothing"}
)
```

### 4. Optimize Time Intelligence

❌ **Poor Performance - Complex Date Logic:**
```dax
YTD Sales = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    FILTER(
        'Dim Calendar',
        'Dim Calendar'[Date] >= DATE(YEAR(TODAY()), 1, 1) &&
        'Dim Calendar'[Date] <= TODAY()
    )
)
```

✅ **Optimized - Built-in Time Intelligence:**
```dax
YTD Sales = TOTALYTD(SUM('Fact Sales'[sales_amount]), 'Dim Calendar'[Date])
```

### 5. Use DIVIDE Instead of Division Operator

❌ **Poor Performance - Division with Error Handling:**
```dax
Average Price = 
IF(
    SUM('Fact Sales'[sales_quantity]) = 0,
    0,
    SUM('Fact Sales'[sales_amount]) / SUM('Fact Sales'[sales_quantity])
)
```

✅ **Optimized - DIVIDE Function:**
```dax
Average Price = 
DIVIDE(
    SUM('Fact Sales'[sales_amount]),
    SUM('Fact Sales'[sales_quantity]),
    0
)
```

### 6. Optimize Context Transition

❌ **Poor Performance - Unnecessary Context Transition:**
```dax
Row Level Calculation = 
SUMX(
    'Fact Sales',
    CALCULATE(SUM('Fact Sales'[sales_amount]))
)
```

✅ **Optimized - Direct Aggregation:**
```dax
Row Level Calculation = SUM('Fact Sales'[sales_amount])
```

---

## Model Size Reduction Strategies

### 1. Remove Unnecessary Columns

**Before Optimization:**
```dax
// Keep only essential columns in fact tables
'Fact Sales'[sales_id]          // Remove - surrogate key not needed
'Fact Sales'[timestamp]         // Remove - use date_key instead
'Fact Sales'[sales_amount]      // Keep - measure source
'Fact Sales'[sales_quantity]    // Keep - measure source
'Fact Sales'[date_key]          // Keep - relationship key
'Fact Sales'[product_id]        // Keep - relationship key
'Fact Sales'[store_id]          // Keep - relationship key
```

**After Optimization:**
```dax
// Optimized fact table with only necessary columns
'Fact Sales'[sales_amount]      // Keep - measure source
'Fact Sales'[sales_quantity]    // Keep - measure source
'Fact Sales'[date_key]          // Keep - relationship key
'Fact Sales'[product_id]        // Keep - relationship key
'Fact Sales'[store_id]          // Keep - relationship key
```

### 2. Optimize Data Types

**Before Optimization:**
```dax
// Inefficient data types
'Dim Product'[product_id]     // Text - "PROD001"
'Dim Product'[price]          // Decimal - high precision
'Dim Product'[is_active]      // Text - "Yes"/"No"
```

**After Optimization:**
```dax
// Optimized data types
'Dim Product'[product_id]     // Whole Number - 1, 2, 3
'Dim Product'[price]          // Fixed Decimal - 2 decimal places
'Dim Product'[is_active]      // True/False - Boolean
```

### 3. Use Calculated Columns Wisely

❌ **Poor Performance - Calculated Column for Aggregation:**
```dax
// Don't create calculated columns for aggregations
Sales Amount in Thousands = 'Fact Sales'[sales_amount] / 1000
```

✅ **Optimized - Measure for Aggregation:**
```dax
// Use measures for aggregations
Sales Amount (Thousands) = SUM('Fact Sales'[sales_amount]) / 1000
```

### 4. Optimize Relationships

**Star Schema (Recommended):**
```
Dim Product ──→ Fact Sales ←── Dim Store
                    │
                    ↓
                Dim Calendar
```

**Avoid Snowflake Schema:**
```
Dim Product Category ──→ Dim Product ──→ Fact Sales
```

### 5. Use Aggregation Tables

**Create Aggregation Table:**
```dax
// Pre-aggregated monthly sales
Monthly Sales Aggregate = 
SUMMARIZE(
    'Fact Sales',
    'Dim Calendar'[Year],
    'Dim Calendar'[Month],
    'Dim Product'[Product Category],
    "Monthly Sales", SUM('Fact Sales'[sales_amount]),
    "Monthly Quantity", SUM('Fact Sales'[sales_quantity])
)
```

---

## Relationship Optimization

### 1. Use Single-Direction Relationships

❌ **Poor Performance - Bi-directional Relationships:**
```
Dim Product ←→ Fact Sales  // Bi-directional
```

✅ **Optimized - Single-Direction:**
```
Dim Product ──→ Fact Sales  // Single-direction
```

### 2. Optimize Cardinality

**Proper Cardinality Settings:**
```
Dim Product (One) ──→ Fact Sales (Many)     // 1:* relationship
Dim Store (One) ──→ Fact Sales (Many)       // 1:* relationship
Dim Calendar (One) ──→ Fact Sales (Many)    // 1:* relationship
```

### 3. Use Integer Keys for Relationships

❌ **Poor Performance - Text Keys:**
```dax
// Text-based keys
'Dim Product'[product_code] = "PROD-001"
'Fact Sales'[product_code] = "PROD-001"
```

✅ **Optimized - Integer Keys:**
```dax
// Integer-based keys
'Dim Product'[product_key] = 1
'Fact Sales'[product_key] = 1
```

### 4. Minimize Role-Playing Dimensions

❌ **Poor Performance - Multiple Date Relationships:**
```
Dim Calendar ──→ Fact Sales (Order Date)
Dim Calendar ──→ Fact Sales (Ship Date)
Dim Calendar ──→ Fact Sales (Delivery Date)
```

✅ **Optimized - Single Active Relationship:**
```
Dim Calendar ──→ Fact Sales (Order Date) [Active]
Dim Calendar ──→ Fact Sales (Ship Date) [Inactive]
Dim Calendar ──→ Fact Sales (Delivery Date) [Inactive]

// Use USERELATIONSHIP for inactive relationships
Ship Date Sales = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    USERELATIONSHIP('Fact Sales'[ship_date], 'Dim Calendar'[Date])
)
```

---

## Query Performance Optimization

### 1. Use Appropriate Filter Context

❌ **Poor Performance - Large Filter Context:**
```dax
Filtered Sales = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    FILTER(
        ALL('Fact Sales'),
        'Fact Sales'[sales_amount] > 1000
    )
)
```

✅ **Optimized - Dimension Filtering:**
```dax
Filtered Sales = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    'Dim Product'[Product Category] = "Electronics"
)
```

### 2. Optimize TOPN Queries

❌ **Poor Performance - TOPN with Complex Logic:**
```dax
Top 10 Products = 
TOPN(
    10,
    ADDCOLUMNS(
        VALUES('Dim Product'[Product]),
        "Sales", CALCULATE(SUM('Fact Sales'[sales_amount]))
    ),
    [Sales],
    DESC
)
```

✅ **Optimized - Simplified TOPN:**
```dax
Top 10 Products = 
TOPN(
    10,
    VALUES('Dim Product'[Product]),
    [Total Sales],
    DESC
)
```

### 3. Use SUMMARIZE Efficiently

❌ **Poor Performance - Complex SUMMARIZE:**
```dax
Product Summary = 
SUMMARIZE(
    'Fact Sales',
    'Dim Product'[Product],
    'Dim Product'[Category],
    'Dim Store'[Store Name],
    "Total Sales", SUM('Fact Sales'[sales_amount]),
    "Avg Price", AVERAGE('Fact Sales'[sales_amount])
)
```

✅ **Optimized - Focused SUMMARIZE:**
```dax
Product Summary = 
SUMMARIZE(
    'Fact Sales',
    'Dim Product'[Product],
    'Dim Product'[Category],
    "Total Sales", SUM('Fact Sales'[sales_amount])
)
```

---

## Memory Management

### 1. Use Calculated Columns vs Measures

**Memory Usage Comparison:**
```dax
// Calculated Column (stored in memory)
Year Column = YEAR('Dim Calendar'[Date])  // 365 rows * 4 bytes = 1.4 KB

// Measure (calculated on-demand)
Year Measure = YEAR(MAX('Dim Calendar'[Date]))  // 0 bytes stored
```

### 2. Optimize String Columns

❌ **Poor Performance - Long Text Columns:**
```dax
// Full product description (high cardinality)
'Dim Product'[Full Description] = "Electronics - Laptop - Dell Inspiron 15 3000 Series..."
```

✅ **Optimized - Categorical Columns:**
```dax
// Category-based description (low cardinality)
'Dim Product'[Category] = "Electronics"
'Dim Product'[Subcategory] = "Laptop"
'Dim Product'[Brand] = "Dell"
```

### 3. Use Compression-Friendly Data

**Optimize for VertiPaq Compression:**
```dax
// Use consistent formatting
'Dim Product'[Product Code] = "PROD-0001"  // Not "PROD-1", "Prod-01", etc.

// Use standardized categories
'Dim Product'[Status] = "Active"  // Not "active", "ACTIVE", "Active ", etc.

// Use integers where possible
'Dim Product'[Priority] = 1  // Not "High", "Medium", "Low"
```

---

## Performance Testing Methodologies

### 1. DAX Query Performance Testing

**Test Query Performance:**
```dax
// Test query with DAX Studio
EVALUATE
ADDCOLUMNS(
    VALUES('Dim Product'[Product]),
    "Sales Amount", [Total Sales],
    "Sales Quantity", [Total Quantity]
)
ORDER BY [Sales Amount] DESC
```

**Performance Metrics to Monitor:**
- Query duration
- Formula engine queries
- Storage engine queries
- Total rows scanned
- Memory usage

### 2. Measure Performance Comparison

**Before Optimization:**
```dax
Slow Measure = 
SUMX(
    'Fact Sales',
    IF(
        RELATED('Dim Product'[Category]) = "Electronics",
        'Fact Sales'[sales_amount],
        0
    )
)
```

**After Optimization:**
```dax
Fast Measure = 
CALCULATE(
    SUM('Fact Sales'[sales_amount]),
    'Dim Product'[Category] = "Electronics"
)
```

**Performance Test Results:**
| Measure | Duration | FE Queries | SE Queries | Memory |
|---------|----------|------------|------------|---------|
| Slow    | 2.5s     | 15         | 8          | 125 MB  |
| Fast    | 0.3s     | 2          | 1          | 25 MB   |

### 3. Load Testing

**Simulate Multiple Users:**
```powershell
# PowerShell script for load testing
$queries = @(
    "EVALUATE VALUES('Dim Product'[Product])",
    "EVALUATE SUMMARIZE('Fact Sales', 'Dim Calendar'[Year], 'Total', SUM('Fact Sales'[sales_amount]))",
    "EVALUATE TOPN(10, VALUES('Dim Product'[Product]), [Total Sales], DESC)"
)

foreach ($query in $queries) {
    $startTime = Get-Date
    # Execute query
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds
    Write-Host "Query: $query - Duration: $duration ms"
}
```

---

## Monitoring and Diagnostics

### 1. Performance Analyzer

**Key Metrics to Monitor:**
- Visual rendering time
- DAX query time
- Visual display time
- Other events

**Optimization Targets:**
- Visual rendering: < 1 second
- DAX query: < 500ms
- Total visual load: < 2 seconds

### 2. DAX Studio Profiling

**Server Timings to Monitor:**
```sql
-- Example server timings from DAX Studio
Duration: 1,234 ms
Query: 234 ms
Formula Engine: 123 ms
Storage Engine: 111 ms
```

**Optimization Priorities:**
1. Storage Engine duration > 50% of total → Optimize relationships/filters
2. Formula Engine duration > 50% of total → Optimize DAX expressions
3. High query count → Reduce context transitions

### 3. Refresh Performance Monitoring

**Monitor Refresh Operations:**
```dax
// Create refresh performance table
Refresh Performance = 
DATATABLE(
    "Table Name", STRING,
    "Refresh Duration", INTEGER,
    "Rows Processed", INTEGER,
    "Memory Used", INTEGER,
    {
        {"Fact Sales", 45, 1000000, 256},
        {"Dim Product", 5, 10000, 32},
        {"Dim Store", 3, 5000, 16}
    }
)
```

---

## Best Practices Summary

### 1. Model Design
- ✅ Use star schema design
- ✅ Minimize model size
- ✅ Use appropriate data types
- ✅ Remove unnecessary columns
- ✅ Use integer keys for relationships

### 2. DAX Optimization
- ✅ Use variables for complex calculations
- ✅ Avoid expensive iterator functions
- ✅ Use built-in time intelligence functions
- ✅ Optimize filter context
- ✅ Use DIVIDE instead of division operator

### 3. Relationship Optimization
- ✅ Use single-direction relationships
- ✅ Minimize bi-directional relationships
- ✅ Use proper cardinality settings
- ✅ Optimize role-playing dimensions

### 4. Query Performance
- ✅ Use appropriate filter context
- ✅ Optimize TOPN queries
- ✅ Use SUMMARIZE efficiently
- ✅ Minimize context transitions

### 5. Memory Management
- ✅ Choose calculated columns vs measures wisely
- ✅ Optimize string columns
- ✅ Use compression-friendly data
- ✅ Monitor memory usage

### 6. Testing and Monitoring
- ✅ Use DAX Studio for performance testing
- ✅ Monitor key performance metrics
- ✅ Implement load testing
- ✅ Use Performance Analyzer
- ✅ Profile refresh operations

---

## Performance Optimization Checklist

### Pre-Optimization Analysis
- [ ] Identify slow-performing visuals
- [ ] Analyze DAX query performance
- [ ] Check model size and memory usage
- [ ] Review relationship design
- [ ] Identify complex measures

### Optimization Implementation
- [ ] Optimize data model structure
- [ ] Refactor slow DAX expressions
- [ ] Optimize relationships
- [ ] Implement aggregation tables
- [ ] Remove unnecessary columns

### Post-Optimization Validation
- [ ] Test visual performance
- [ ] Validate query performance
- [ ] Check memory usage
- [ ] Verify functionality
- [ ] Document optimizations

### Ongoing Monitoring
- [ ] Regular performance reviews
- [ ] Monitor refresh performance
- [ ] Track user experience metrics
- [ ] Update optimization strategies
- [ ] Maintain documentation

---

## Tools and Resources

### Essential Tools
- **DAX Studio**: Query performance analysis
- **Tabular Editor**: Model optimization
- **Performance Analyzer**: Visual performance monitoring
- **ALM Toolkit**: Model comparison
- **Power BI Premium Metrics**: Usage monitoring

### Performance Monitoring Queries
```dax
// Model size analysis
Model Size = 
SUMMARIZE(
    INFO.TABLES(),
    [TableName],
    "Rows", [RowsCount],
    "Size", [Size],
    "Compression", DIVIDE([Size], [UncompressedSize])
)

// Relationship analysis
Relationship Performance = 
SUMMARIZE(
    INFO.RELATIONSHIPS(),
    [FromTable],
    [ToTable],
    "Cardinality", [Cardinality],
    "Cross Filter", [CrossFilterDirection]
)
```

### Performance Baselines
- **Model refresh**: < 10 minutes for 1M rows
- **Visual loading**: < 2 seconds per visual
- **DAX query**: < 500ms per query
- **Report navigation**: < 1 second per page
- **Model size**: < 1GB for optimal performance 