@echo off
echo "=== Modern Data Stack Showcase - Master Model Deployment ==="
echo.

echo "Step 1: Create master model.bim from TMDL source"
start /wait /d "c:\Program Files (x86)\Tabular Editor" TabularEditor.exe "%~dp0..\model-folder" -B "%~dp0..\temp\Modern Data Stack Master.Dataset\model.bim"

echo "Step 2: Create grocery connector model TMDL"
start /wait /d "c:\Program Files (x86)\Tabular Editor" TabularEditor.exe "%~dp0..\temp\Modern Data Stack Master.Dataset\model.bim" -S "%~dp0child-models\$grocery_connector.csx" -TMDL "%~dp0..\..\grocery-connector\model-folder"

echo "Step 3: Create retail connector model TMDL"
start /wait /d "c:\Program Files (x86)\Tabular Editor" TabularEditor.exe "%~dp0..\temp\Modern Data Stack Master.Dataset\model.bim" -S "%~dp0child-models\$retail_connector.csx" -TMDL "%~dp0..\..\retail-connector\model-folder"

echo "Step 4: Create harmonized cross-connector model TMDL"
start /wait /d "c:\Program Files (x86)\Tabular Editor" TabularEditor.exe "%~dp0..\temp\Modern Data Stack Master.Dataset\model.bim" -S "%~dp0child-models\$harmonized.csx" -TMDL "%~dp0..\..\harmonized\model-folder"

echo "Step 5: Clean up temporary files"
del "%~dp0..\temp\Modern Data Stack Master.Dataset\model.bim"
rmdir /s /q "%~dp0..\temp"

echo.
echo "=== Deployment Complete ==="
echo "Generated models:"
echo "  - Grocery Connector: grocery-connector\model-folder"
echo "  - Retail Connector: retail-connector\model-folder" 
echo "  - Harmonized: harmonized\model-folder"
echo.
pause 