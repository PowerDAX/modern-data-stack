# Standard DAX Measure Templates

## Table of Contents
1. [Basic Aggregations](#basic-aggregations)
2. [Time Intelligence](#time-intelligence)
3. [Variance Analysis](#variance-analysis)
4. [Growth Calculations](#growth-calculations)
5. [Ranking and Top N](#ranking-and-top-n)
6. [Percentage Calculations](#percentage-calculations)
7. [Performance Metrics](#performance-metrics)
8. [Data Quality Metrics](#data-quality-metrics)
9. [Financial Calculations](#financial-calculations)
10. [Utility Measures](#utility-measures)

---

## Basic Aggregations

### Total Amount
```dax
Total Amount = SUM(FactTable[Amount])
```

### Average Amount
```dax
Average Amount = AVERAGE(FactTable[Amount])
```

### Count of Records
```dax
Record Count = COUNTROWS(FactTable)
```

### Count of Distinct Values
```dax
Distinct Count = DISTINCTCOUNT(FactTable[Field])
```

### Max Value
```dax
Max Amount = MAX(FactTable[Amount])
```

### Min Value
```dax
Min Amount = MIN(FactTable[Amount])
```

---

## Time Intelligence

### Year-to-Date (YTD)
```dax
YTD Amount = TOTALYTD(SUM(FactTable[Amount]), 'Dim Calendar'[Date])
```

### Month-to-Date (MTD)
```dax
MTD Amount = TOTALMTD(SUM(FactTable[Amount]), 'Dim Calendar'[Date])
```

### Quarter-to-Date (QTD)
```dax
QTD Amount = TOTALQTD(SUM(FactTable[Amount]), 'Dim Calendar'[Date])
```

### Previous Year
```dax
Previous Year Amount = CALCULATE(
    SUM(FactTable[Amount]),
    SAMEPERIODLASTYEAR('Dim Calendar'[Date])
)
```

### Previous Month
```dax
Previous Month Amount = CALCULATE(
    SUM(FactTable[Amount]),
    DATEADD('Dim Calendar'[Date], -1, MONTH)
)
```

### Previous Quarter
```dax
Previous Quarter Amount = CALCULATE(
    SUM(FactTable[Amount]),
    DATEADD('Dim Calendar'[Date], -1, QUARTER)
)
```

### Rolling 12 Months
```dax
Rolling 12 Months Amount = CALCULATE(
    SUM(FactTable[Amount]),
    DATESINPERIOD(
        'Dim Calendar'[Date],
        LASTDATE('Dim Calendar'[Date]),
        -12,
        MONTH
    )
)
```

### Rolling 90 Days
```dax
Rolling 90 Days Amount = CALCULATE(
    SUM(FactTable[Amount]),
    DATESINPERIOD(
        'Dim Calendar'[Date],
        LASTDATE('Dim Calendar'[Date]),
        -90,
        DAY
    )
)
```

---

## Variance Analysis

### Variance vs Previous Year
```dax
Variance vs PY = [Total Amount] - [Previous Year Amount]
```

### Variance % vs Previous Year
```dax
Variance % vs PY = 
DIVIDE(
    [Total Amount] - [Previous Year Amount],
    [Previous Year Amount],
    0
)
```

### Variance vs Budget
```dax
Variance vs Budget = [Total Amount] - [Budget Amount]
```

### Variance % vs Budget
```dax
Variance % vs Budget = 
DIVIDE(
    [Total Amount] - [Budget Amount],
    [Budget Amount],
    0
)
```

---

## Growth Calculations

### Year-over-Year Growth
```dax
YoY Growth = 
DIVIDE(
    [Total Amount] - [Previous Year Amount],
    [Previous Year Amount],
    0
)
```

### Month-over-Month Growth
```dax
MoM Growth = 
DIVIDE(
    [Total Amount] - [Previous Month Amount],
    [Previous Month Amount],
    0
)
```

### Compound Annual Growth Rate (CAGR)
```dax
CAGR = 
VAR StartValue = [First Year Amount]
VAR EndValue = [Last Year Amount]
VAR Years = [Number of Years]
RETURN
DIVIDE(
    POWER(DIVIDE(EndValue, StartValue), DIVIDE(1, Years)) - 1,
    1,
    0
)
```

---

## Ranking and Top N

### Current Rank
```dax
Current Rank = RANKX(
    ALL(DimTable[Field]),
    [Total Amount],
    ,
    DESC
)
```

### Top 10 Flag
```dax
Top 10 Flag = IF([Current Rank] <= 10, "Top 10", "Other")
```

### Percentile Rank
```dax
Percentile Rank = PERCENTRANK.INC(
    ALL(DimTable[Field]),
    [Total Amount]
)
```

### Running Total
```dax
Running Total = CALCULATE(
    SUM(FactTable[Amount]),
    FILTER(
        ALL('Dim Calendar'[Date]),
        'Dim Calendar'[Date] <= MAX('Dim Calendar'[Date])
    )
)
```

---

## Percentage Calculations

### Percentage of Total
```dax
% of Total = DIVIDE(
    [Total Amount],
    CALCULATE([Total Amount], ALL(DimTable)),
    0
)
```

### Percentage of Parent
```dax
% of Parent = DIVIDE(
    [Total Amount],
    CALCULATE([Total Amount], ALLEXCEPT(DimTable, DimTable[ParentField])),
    0
)
```

### Percentage of Grand Total
```dax
% of Grand Total = DIVIDE(
    [Total Amount],
    CALCULATE([Total Amount], ALLSELECTED()),
    0
)
)
```

---

## Performance Metrics

### Average Selling Price
```dax
Average Selling Price = DIVIDE(
    [Total Sales Amount],
    [Total Sales Quantity],
    0
)
```

### Conversion Rate
```dax
Conversion Rate = DIVIDE(
    [Total Conversions],
    [Total Visits],
    0
)
```

### Return on Investment (ROI)
```dax
ROI = DIVIDE(
    [Total Revenue] - [Total Cost],
    [Total Cost],
    0
)
```

### Profit Margin
```dax
Profit Margin = DIVIDE(
    [Total Revenue] - [Total Cost],
    [Total Revenue],
    0
)
```

---

## Data Quality Metrics

### Completion Rate
```dax
Completion Rate = DIVIDE(
    COUNTROWS(FILTER(FactTable, NOT(ISBLANK(FactTable[Field])))),
    COUNTROWS(FactTable),
    0
)
```

### Data Freshness (Days)
```dax
Data Freshness = DATEDIFF(
    MAX(FactTable[LoadDate]),
    TODAY(),
    DAY
)
```

### Error Rate
```dax
Error Rate = DIVIDE(
    COUNTROWS(FILTER(FactTable, FactTable[ErrorFlag] = TRUE())),
    COUNTROWS(FactTable),
    0
)
```

---

## Financial Calculations

### Net Present Value (NPV)
```dax
NPV = 
SUMX(
    VALUES('Dim Calendar'[Year]),
    DIVIDE(
        CALCULATE([Cash Flow]),
        POWER(1 + [Discount Rate], 'Dim Calendar'[Year] - [Base Year])
    )
)
```

### Internal Rate of Return (IRR)
```dax
IRR = 
// Complex calculation requiring iterative approach
// Use Power Query or external calculation for precision
```

### Payback Period
```dax
Payback Period = 
VAR CumulativeCashFlow = [Running Total Cash Flow]
VAR InitialInvestment = [Initial Investment]
RETURN
CALCULATE(
    MIN('Dim Calendar'[Year]),
    FILTER(
        ALL('Dim Calendar'[Year]),
        CumulativeCashFlow >= InitialInvestment
    )
)
```

---

## Utility Measures

### Selected Value Helper
```dax
Selected Value = 
IF(
    HASONEVALUE(DimTable[Field]),
    VALUES(DimTable[Field]),
    "Multiple Values"
)
```

### Dynamic Title
```dax
Dynamic Title = 
"Analysis for " & [Selected Value]
```

### Format Helper
```dax
Format Helper = 
SWITCH(
    TRUE(),
    [Total Amount] >= 1000000, FORMAT([Total Amount]/1000000, "#,##0.0") & "M",
    [Total Amount] >= 1000, FORMAT([Total Amount]/1000, "#,##0.0") & "K",
    FORMAT([Total Amount], "#,##0")
)
```

### Color Helper
```dax
Color Helper = 
SWITCH(
    TRUE(),
    [Variance % vs PY] > 0.1, "Green",
    [Variance % vs PY] < -0.1, "Red",
    "Gray"
)
```

### Tooltip Helper
```dax
Tooltip Helper = 
"Current: " & FORMAT([Total Amount], "#,##0") & 
UNICHAR(10) & 
"Previous Year: " & FORMAT([Previous Year Amount], "#,##0") & 
UNICHAR(10) & 
"Change: " & FORMAT([Variance % vs PY], "0.0%")
```

---

## Usage Guidelines

### Formatting Standards
- **Currency**: `$ #,##0;($ #,##0)`
- **Percentage**: `#,##0.00 %;(#,##0.00 %)`
- **Integer**: `#,##0;(#,##0)`
- **Decimal**: `#,##0.00;(#,##0.00)`

### Performance Best Practices
1. Use variables (VAR) for complex calculations
2. Avoid iterating functions in measures when possible
3. Use DIVIDE() instead of / for division to handle zeros
4. Consider using SELECTEDVALUE() for single value contexts
5. Use ISBLANK() and HASONEVALUE() for conditional logic

### Naming Conventions
- Use descriptive names with proper case
- Include units when applicable (e.g., "Sales Amount ($)")
- Use consistent prefixes for related measures
- Avoid abbreviations unless widely understood

### Display Folder Organization
- **Measures**: All standard measures
- **Time Intelligence**: All time-based calculations
- **Variance Analysis**: All variance measures
- **Percentages**: All percentage calculations
- **Utilities**: Helper measures and formatting

---

## Template Usage Instructions

1. **Copy the template**: Select the appropriate template for your use case
2. **Replace table references**: Update `FactTable` and `DimTable` with your actual table names
3. **Update field references**: Replace `[Field]` and `[Amount]` with your actual column names
4. **Apply formatting**: Use the standard formatting strings provided
5. **Set display folder**: Organize measures in appropriate display folders
6. **Test and validate**: Ensure calculations work correctly with your data model

---

## Advanced Patterns

### Dynamic Measure Selection
```dax
Dynamic Measure = 
SWITCH(
    SELECTEDVALUE(MeasureTable[MeasureType]),
    "Revenue", [Total Revenue],
    "Profit", [Total Profit],
    "Units", [Total Units],
    [Total Revenue]
)
```

### Conditional Aggregation
```dax
Conditional Sum = 
SUMX(
    FactTable,
    IF(
        FactTable[Category] = "A",
        FactTable[Amount] * 1.1,
        FactTable[Amount]
    )
)
```

### Cross-Table Calculations
```dax
Cross Table Calculation = 
SUMX(
    RELATEDTABLE(FactTable),
    FactTable[Amount] * RELATED(DimTable[Rate])
)
``` 