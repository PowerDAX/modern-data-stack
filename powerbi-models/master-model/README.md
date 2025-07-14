# Modern Data Stack Master Model

This implements the **Master Model Pattern** for Power BI as described in the [Tabular Editor documentation](https://docs.tabulareditor.com/te2/Master-model-pattern.html). This pattern allows you to maintain a single master model that can be deployed as multiple connector-specific child models.

## Overview

The master model contains all tables, measures, and business logic for the modern data stack showcase. It uses **perspectives** to define how the model should be split into connector-specific versions:

- **$harmonized** - Cross-connector harmonized model
- **$grocery_connector** - Grocery-specific model with item/location nomenclature  
- **$retail_connector** - Retail-specific model with product/store nomenclature

## Directory Structure

```
master-model/
├── model-folder/           # TMDL master model files
│   ├── database.tmdl       # Database definition
│   ├── model.tmdl          # Model with perspectives
│   ├── expressions.tmdl    # Shared expressions
│   ├── relationships.tmdl  # Star schema relationships
│   ├── perspectives/       # Perspective definitions
│   │   ├── $harmonized.tmdl
│   │   ├── $grocery_connector.tmdl
│   │   └── $retail_connector.tmdl
│   └── tables/             # Table definitions
│       ├── Fact Sales.tmdl
│       ├── Fact Inventory.tmdl
│       ├── Dim Product.tmdl
│       ├── Dim Store.tmdl
│       ├── Dim Calendar.tmdl
│       └── Dim Time Period.tmdl
├── scripts/                # Child model generation scripts
│   ├── deploy-child-models.ps1
│   ├── deploy-child-models.bat
│   └── child-models/
│       ├── $harmonized.csx
│       ├── $grocery_connector.csx
│       └── $retail_connector.csx
└── README.md
```

## Key Concepts

### 1. Developer Perspectives

Perspectives prefixed with `$` are used by developers to explicitly define which objects belong to each model:

- `$harmonized` - Cross-connector harmonized model with standardized nomenclature
- `$grocery_connector` - Grocery-specific model with Item/Location nomenclature and promotional sales
- `$retail_connector` - Retail-specific model with Product/Store nomenclature, excluding grocery-only features

Each perspective explicitly lists the tables, measures, and columns that should be included in that connector's model. Objects not included in a perspective are automatically excluded from the generated child model.

### 2. Systematic Nomenclature Transformation

The grocery connector uses an intelligent transformation system that automatically converts Product→Item and Store→Location nomenclature across all objects:

**Table Transformations:**
- `Dim Product` → `Dim Item`
- `Dim Store` → `Dim Location`  
- `Fact Inventory` → `Fact Inventory Location`

**Column Transformations:**
- `Product Brand` → `Item Brand`
- `Store Name` → `Location Name`
- `product_id` → `item_id`
- `store_key` → `location_key`

**Measure Expression Updates:**
```dax
// Original (Master/Retail)
SUM('Dim Product'[Product Brand])

// Automatically becomes (Grocery)  
SUM('Dim Item'[Item Brand])
```

**Custom Annotations Override:**
For specific column renaming needs beyond the systematic transformation:

```tmdl
column sales_amount
    dataType: decimal
    summarizeBy: sum
    annotation '$grocery_connector_ColumnName' = "sales_dollars"
```

### 3. Measure Library

Each fact table includes a comprehensive measure library:

**Fact Sales Measures:**
- `Sales Amount`, `Sales Quantity` - Core sales metrics
- `Regular Sales Amount/Quantity` - Grocery-specific regular sales
- `Promotional Sales Amount/Quantity` - Grocery-specific promotional sales
- `Average Selling Price` - Calculated measure for price analytics
- `Total Sales Amount/Quantity (Grocery)` - Combined regular + promotional
- `Promotional Sales %` - Promotional sales percentage
- `Transaction Count`, `Product Count`, `Store Count` - Distinct count measures

**Fact Inventory Measures:**
- `On Hand Quantity/Amount` - Current inventory levels
- `On Order Quantity/Amount` - Grocery-specific incoming inventory
- `Allocated Quantity`, `Available Quantity` - Grocery-specific allocations
- `Total Inventory Value` - Combined on hand + on order value
- `Available for Sale Quantity` - On hand minus allocated
- `Inventory Utilization %` - Allocation percentage
- `Average Inventory Value` - Price per unit calculation
- `Inventory Record Count`, `Product Count`, `Store Count` - Distinct count measures

**Measure Formatting Standards:**
- All measures are organized in the "Measures" display folder
- Integer values: `#,##0;(#,##0)`
- Dollar amounts: `$ #,##0.00;($ #,##0.00)`
- Percentages: `#,##0.00 %;(#,##0.00 %)`

### 4. Measure Inclusion by Perspective

Measures are included in child models based on their perspective membership. Each perspective explicitly defines which measures are included:

```tmdl
// In $grocery_connector.tmdl perspective:
perspectiveTable 'Fact Sales'
    perspectiveMeasure 'Regular Sales Amount'
    perspectiveMeasure 'Promotional Sales Amount'
    perspectiveMeasure 'Total Sales Amount (Grocery)'

// In $retail_connector.tmdl perspective:  
perspectiveTable 'Fact Sales'
    perspectiveMeasure 'Sales Amount'
    perspectiveMeasure 'Sales Quantity'
    // Regular/Promotional measures excluded
```

The child model generation script removes objects not included in the perspective: `if(!m.InPerspective[version]) m.Delete()`

### 5. Table Source Control

Each table has annotations for connector-specific data sources:

```tmdl
annotation '$harmonized.BigQuery.TableId' = 'modern-data-stack-showcase.dbt_analytics.analytics_harmonized_fact_sales'
annotation '$grocery_connector.BigQuery.TableId' = 'modern-data-stack-showcase.dbt_analytics.analytics_grocery_connector_fact_sales'  
annotation '$retail_connector.BigQuery.TableId' = 'modern-data-stack-showcase.dbt_analytics.analytics_retail_connector_fact_sales'
```

## Connector-Specific Features

### Harmonized Model
- Cross-connector standardized schema
- All tables use consistent naming (product, store, sales_amount, sales_quantity)
- Includes all measures but uses harmonized expressions

### Grocery Connector Model  
- **Systematic Nomenclature Transformation**: Product → Item, Store → Location
- **Table Renaming**: `Dim Product` → `Dim Item`, `Dim Store` → `Dim Location`, `Fact Inventory` → `Fact Inventory Location`
- **Column Renaming**: All Product/product columns become Item/item, all Store/store columns become Location/location
- **Measure Expression Updates**: All DAX expressions automatically updated to use transformed table and column names
- Includes grocery-specific measures for regular vs promotional sales
- Additional columns: `regular_sales_amount`, `regular_sales_quantity`, `promotional_sales_amount`, `promotional_sales_quantity`
- Inventory includes grocery-specific columns: `available_quantity`, `allocated_quantity`, `on_order_quantity`, `on_order_amount`

### Retail Connector Model
- Uses retail nomenclature: **product** and **store** (standard naming)
- Grocery-specific promotional/regular sales columns and measures are removed
- Grocery-specific inventory columns are excluded (`available_quantity`, `allocated_quantity`, `allocated_units`, `available_units`)
- Simplified fact structure matching retail connector data

## Usage Instructions

### 1. Open Master Model in Tabular Editor

```bash
# Open the model in Tabular Editor 3
TabularEditor3.exe "model-folder"
```

### 2. Working with Perspectives

**View Current Perspectives:**
- Switch between `$harmonized`, `$grocery_connector`, `$retail_connector` perspectives in Tabular Editor
- Each perspective shows only the objects included in that specific model

**Adding Objects to Perspectives:**
1. Select tables, measures, or columns in the model explorer
2. Right-click and select "Add to Perspective" 
3. Choose the appropriate developer perspective(s)
4. Verify object visibility by switching between perspectives

**Removing Objects from Perspectives:**
1. Select objects in the model explorer
2. Right-click and select "Remove from Perspective"
3. Choose the perspective to remove from
4. Verify removal by switching to that perspective

### 3. Deploy Child Models

Use the automated deployment scripts to generate all connector-specific models:

**Option 1: PowerShell Script (Recommended)**
```powershell
# Generate all child models as TMDL
.\scripts\deploy-child-models.ps1

# With custom Tabular Editor path
.\scripts\deploy-child-models.ps1 -TabularEditorPath "C:\Tools\TabularEditor\TabularEditor.exe"
```

**Option 2: Windows Batch Script**
```batch
.\scripts\deploy-child-models.bat
```

**Option 3: Manual Deployment**
```bash
# Deploy harmonized model
TabularEditor3.exe "model-folder" -S "scripts/child-models/$harmonized.csx" -D "server" "ModernDataStackHarmonized" -O -R

# Deploy grocery connector model  
TabularEditor3.exe "model-folder" -S "scripts/child-models/$grocery_connector.csx" -D "server" "ModernDataStackGrocery" -O -R

# Deploy retail connector model
TabularEditor3.exe "model-folder" -S "scripts/child-models/$retail_connector.csx" -D "server" "ModernDataStackRetail" -O -R
```

The deployment scripts will create the following structure:
```
powerbi-models/
├── master-model/              # This master model
├── grocery-connector/         # Generated with Item/Location nomenclature
│   └── model-folder/          
├── retail-connector/          # Generated with Product/Store nomenclature
│   └── model-folder/          
└── harmonized/                # Generated cross-connector model
    └── model-folder/          
```

## Development Workflow

### Adding New Tables

1. Create the table TMDL file with all possible columns
2. Add table to appropriate perspectives using Tabular Editor
3. Include necessary columns and measures in each perspective
4. Test by switching between perspectives to verify inclusion/exclusion

### Adding New Measures

1. Define measure with harmonized expression  
2. Add measure to appropriate perspectives
3. Test measure in each perspective
4. Deploy to validate functionality

### Modifying Existing Objects

1. Update master model TMDL files
2. Update perspective assignments if needed
3. Test in Tabular Editor with perspective switching
4. Deploy updated child models

## Best Practices

1. **Use Perspectives for Object Inclusion** - Explicitly define which objects belong to each connector using perspectives
2. **Test Perspective Switching** - Verify model works correctly in each perspective before deployment
3. **Leverage Systematic Transformations** - The grocery connector automatically handles Product→Item and Store→Location renaming
4. **Use Custom Annotations Sparingly** - Only override the systematic transformation when specific business rules require it
5. **Version Control** - Use git to track changes to TMDL files and scripts
6. **Automate Deployment** - Use CI/CD pipelines to deploy child models consistently

## Benefits of This Approach

### ✅ **Advantages over Translation Layer:**
- **Single Source of Truth**: All logic in one master model
- **Automatic Consistency**: Systematic transformations ensure no missed renamings
- **DAX Expression Safety**: All measure expressions automatically updated
- **Performance**: No runtime translation overhead
- **Maintainability**: Changes to master model automatically propagate

### ✅ **Developer Experience:**
- **Write Once, Deploy Everywhere**: Define objects once with universal naming
- **Intelligent Transformations**: Product/Store → Item/Location happens automatically
- **Override When Needed**: Custom annotations for special cases
- **Error Prevention**: No manual find-and-replace errors

### ✅ **Perspective-Based Benefits:**
- **Explicit Object Control**: Clear visibility of what's included in each model
- **No Annotation Complexity**: Simple perspective membership determines inclusion
- **Dependency Management**: Automatic handling of object dependencies
- **Clean Architecture**: Separated concerns without complex filtering logic

## Troubleshooting

### Missing Objects in Child Model
- Check perspective assignments in master model
- Verify object is included in the correct perspective
- Ensure all dependencies are also included in the perspective

### Incorrect Column Names
- Check column name annotations for custom overrides
- Verify child model script column renaming logic
- Test with correct connector-specific data source

### Measure Expression Errors  
- Validate DAX expressions in each perspective
- Check measure annotations for typos
- Test measure with perspective-specific column names

## New Dimension Tables

### Dim Calendar
Comprehensive date dimension table with Power Query M source:
- **Date Range**: 2020-2030 (configurable)
- **Week Start**: Monday (configurable)
- **Key Attributes**: date_key, Date, Year, Month Name, Week Start/End, Day of Week, Quarter, Fiscal Year
- **Advanced Features**: Holiday flags, weekend identification, fiscal calendar support
- **Performance**: Optimized for time intelligence functions

### Dim Time Period
Dynamic time period dimension based on current date:
- **Base Date**: Today (real-time)
- **Week Logic**: Monday start, last N weeks exclude current week
- **Periods**: Current/Last Week, Last 2/4/13/26/52 weeks, YTD, MTD, QTD, Rolling periods
- **Attributes**: Start/End dates, days count, period category, sort order
- **Use Cases**: Relative time analysis, period-over-period comparisons

### Star Schema Relationships
Complete relationship model with optimized performance:
- **Fact to Dimensions**: One-to-many relationships with proper cardinality
- **Date Integration**: Active relationship from facts to Dim Calendar
- **Time Period Support**: Inactive relationships for flexible time analysis
- **Performance**: Single-direction relationships for optimal query performance

## Completed Features

### ✅ **Phase 3 Complete - Advanced Power BI Models**
- **Master Model**: 6-table star schema with comprehensive measures (23 total)
- **Perspective-Based Architecture**: Clean separation using developer perspectives
- **Shared Assets**: Professional theme, DAX templates, performance optimization guides
- **Azure DevOps**: Complete CI/CD pipeline templates for automated deployment
- **Child Models**: Systematic nomenclature transformation with automated deployment
- **Documentation**: Complete usage guides, troubleshooting, and best practices

### ✅ **Advanced Time Intelligence**
- **Built-in Support**: Optimized for standard time intelligence functions
- **Flexible Periods**: Dynamic time periods with configurable base dates
- **Monday Week Start**: Consistent week definition across all date calculations
- **Fiscal Calendar**: Configurable fiscal year support with quarters and months

### ✅ **Performance Optimization**
- **DAX Best Practices**: Variable usage, DIVIDE function, optimized context transitions
- **Model Optimization**: Integer keys, star schema, minimal cardinality
- **Query Performance**: Indexed relationships, proper data types, compression-friendly data
- **Monitoring**: Performance testing methodologies and diagnostic tools

### ✅ **Enterprise DevOps**
- **Automated Deployment**: PowerShell and batch scripts for child model generation
- **CI/CD Integration**: Azure DevOps pipeline templates with validation and testing
- **Multi-Environment**: Development, staging, production deployment workflows
- **Quality Assurance**: Model validation, DAX syntax checking, performance testing

## Future Enhancements

1. **Additional Dimensions** - Customer, territory, product hierarchies
2. **Advanced Analytics** - ML integration, predictive measures, statistical functions
3. **Data Governance** - Lineage tracking, data quality metrics, audit trails
4. **Integration Patterns** - Real-time data, streaming analytics, external data sources
5. **User Experience** - Custom visuals, report themes, mobile optimization 