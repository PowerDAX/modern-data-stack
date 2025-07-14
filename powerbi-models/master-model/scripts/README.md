# Master Model Deployment Scripts

This directory contains scripts to deploy child models from the Modern Data Stack Showcase master model.

## Overview

The master model pattern allows you to maintain a single source of truth while generating connector-specific models with different nomenclature and object inclusions.

## Generated Models

| Model | Description | Tables | Nomenclature |
|-------|-------------|--------|--------------|
| **Grocery Connector** | Grocery-specific model with all measures | Dim Item, Dim Location, Fact Sales, Fact Inventory Location | Item/Location |
| **Retail Connector** | Retail-specific model, excludes grocery measures | Dim Product, Dim Store, Fact Sales, Fact Inventory Store | Product/Store |
| **Harmonized** | Cross-connector model, common measures only | Dim Product, Dim Store, Fact Sales, Fact Inventory | Product/Store |

## Prerequisites

- **Tabular Editor** installed at `C:\Program Files (x86)\Tabular Editor\TabularEditor.exe`
- Or specify custom path using PowerShell script parameter

## Usage

### Option 1: Batch Script (Windows)
```batch
.\deploy-child-models.bat
```

### Option 2: PowerShell Script (Cross-platform)
```powershell
# Default Tabular Editor path
.\deploy-child-models.ps1

# Custom Tabular Editor path
.\deploy-child-models.ps1 -TabularEditorPath "C:\Tools\TabularEditor\TabularEditor.exe"
```

## Output Structure

```
powerbi-models/
├── master-model/              # Source master model
│   ├── model-folder/          # TMDL source files
│   └── scripts/               # This directory
├── grocery-connector/         # Generated grocery model
│   └── model-folder/          # TMDL with Item/Location nomenclature
├── retail-connector/          # Generated retail model
│   └── model-folder/          # TMDL with Product/Store nomenclature
└── harmonized/                # Generated harmonized model
    └── model-folder/          # TMDL with common objects only
```

## How It Works

1. **Master Model → BIM**: Converts TMDL master model to .bim format
2. **Apply Child Scripts**: Runs each child model transformation script:
   - `$grocery_connector.csx` - Transforms to Item/Location nomenclature
   - `$retail_connector.csx` - Excludes grocery-specific objects  
   - `$harmonized.csx` - Creates cross-connector view
3. **Serialize to TMDL**: Converts each transformed model back to TMDL format
4. **Cleanup**: Removes temporary .bim files

## Child Model Transformations

### Grocery Connector
- **Tables**: `Dim Product` → `Dim Item`, `Dim Store` → `Dim Location`
- **Columns**: All product/store references → item/location
- **Measures**: Includes all grocery-specific measures (Regular Sales, Promotional Sales, etc.)
- **Relationships**: Updated to use transformed table names
- **Source Columns**: Maps to grocery connector analytics tables

### Retail Connector  
- **Tables**: `Fact Inventory` → `Fact Inventory Store`
- **Measures**: Excludes grocery-specific measures
- **Columns**: Excludes grocery-specific columns (regular_sales_amount, allocated_quantity, etc.)

### Harmonized
- **Objects**: Only common measures and columns across connectors
- **Tables**: Standard Product/Store nomenclature
- **Purpose**: Cross-connector analysis and reporting

## Troubleshooting

### Common Issues

1. **Tabular Editor not found**
   - Install Tabular Editor from [https://tabulareditor.com/](https://tabulareditor.com/)
   - Or specify correct path with `-TabularEditorPath` parameter

2. **Permission errors**
   - Run PowerShell as Administrator
   - Ensure write permissions to output directories

3. **Script execution policy (PowerShell)**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Debug Mode

For detailed error information, run the PowerShell script with verbose output:
```powershell
.\deploy-child-models.ps1 -Verbose
``` 