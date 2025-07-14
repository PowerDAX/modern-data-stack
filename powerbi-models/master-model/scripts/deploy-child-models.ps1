#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy child models from Modern Data Stack Showcase master model

.DESCRIPTION
    This script takes the master model TMDL source and generates three child models:
    - Grocery Connector (with Item/Location nomenclature)
    - Retail Connector (with Product/Store nomenclature) 
    - Harmonized (cross-connector model)

.PARAMETER TabularEditorPath
    Path to TabularEditor.exe (defaults to standard installation path)

.EXAMPLE
    .\deploy-child-models.ps1
    
.EXAMPLE
    .\deploy-child-models.ps1 -TabularEditorPath "C:\Tools\TabularEditor\TabularEditor.exe"
#>

param(
    [string]$TabularEditorPath = "C:\Program Files (x86)\Tabular Editor\TabularEditor.exe"
)

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MasterModelDir = Join-Path $ScriptDir "..\model-folder"
$TempDir = Join-Path $ScriptDir "..\temp"
$TempModelPath = Join-Path $TempDir "Modern Data Stack Master.Dataset\model.bim"
$ChildScriptsDir = Join-Path $ScriptDir "child-models"

# Output directories (relative to powerbi-models directory)
$PowerBIModelsDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$GroceryConnectorDir = Join-Path $PowerBIModelsDir "grocery-connector\model-folder"
$RetailConnectorDir = Join-Path $PowerBIModelsDir "retail-connector\model-folder"
$HarmonizedDir = Join-Path $PowerBIModelsDir "harmonized\model-folder"

Write-Host "=== Modern Data Stack Showcase - Master Model Deployment ===" -ForegroundColor Cyan
Write-Host ""

# Validate Tabular Editor exists
if (-not (Test-Path $TabularEditorPath)) {
    Write-Error "Tabular Editor not found at: $TabularEditorPath"
    Write-Host "Please install Tabular Editor or specify the correct path with -TabularEditorPath parameter"
    exit 1
}

# Create temp directory
Write-Host "Step 1: Creating temporary directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $TempModelPath) -Force | Out-Null

try {
    # Step 1: Create master model.bim from TMDL source
    Write-Host "Step 2: Create master model.bim from TMDL source..." -ForegroundColor Yellow
    $process = Start-Process -FilePath $TabularEditorPath -ArgumentList "`"$MasterModelDir`" -B `"$TempModelPath`"" -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Failed to create master model.bim (Exit code: $($process.ExitCode))"
    }

    # Step 2: Create grocery connector model
    Write-Host "Step 3: Create grocery connector model TMDL..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $GroceryConnectorDir -Force | Out-Null
    $groceryScript = Join-Path $ChildScriptsDir "`$grocery_connector.csx"
    $process = Start-Process -FilePath $TabularEditorPath -ArgumentList "`"$TempModelPath`" -S `"$groceryScript`" -TMDL `"$GroceryConnectorDir`"" -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Failed to create grocery connector model (Exit code: $($process.ExitCode))"
    }

    # Step 3: Create retail connector model  
    Write-Host "Step 4: Create retail connector model TMDL..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $RetailConnectorDir -Force | Out-Null
    $retailScript = Join-Path $ChildScriptsDir "`$retail_connector.csx"
    $process = Start-Process -FilePath $TabularEditorPath -ArgumentList "`"$TempModelPath`" -S `"$retailScript`" -TMDL `"$RetailConnectorDir`"" -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Failed to create retail connector model (Exit code: $($process.ExitCode))"
    }

    # Step 4: Create harmonized cross-connector model
    Write-Host "Step 5: Create harmonized cross-connector model TMDL..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $HarmonizedDir -Force | Out-Null
    $harmonizedScript = Join-Path $ChildScriptsDir "`$harmonized.csx"
    $process = Start-Process -FilePath $TabularEditorPath -ArgumentList "`"$TempModelPath`" -S `"$harmonizedScript`" -TMDL `"$HarmonizedDir`"" -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Failed to create harmonized model (Exit code: $($process.ExitCode))"
    }

    Write-Host ""
    Write-Host "=== Deployment Complete ===" -ForegroundColor Green
    Write-Host "Generated models:" -ForegroundColor Green
    Write-Host "  - Grocery Connector: $GroceryConnectorDir" -ForegroundColor White
    Write-Host "  - Retail Connector: $RetailConnectorDir" -ForegroundColor White
    Write-Host "  - Harmonized: $HarmonizedDir" -ForegroundColor White
    Write-Host ""

} catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    exit 1
} finally {
    # Step 5: Clean up temporary files
    Write-Host "Step 6: Cleaning up temporary files..." -ForegroundColor Yellow
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force
    }
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 