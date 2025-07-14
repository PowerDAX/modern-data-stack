# Script to run dbt with mock data
# Usage: ./run_with_mock_data.ps1 [dbt_command]

Write-Host "🔧 Setting up mock data environment..." -ForegroundColor Green

# Set environment variable
$env:USE_MOCK_DATA = 'true'

# Seed mock data first
Write-Host "📦 Seeding mock data..." -ForegroundColor Blue
dbt seed --select mock

# Check if specific dbt command was provided
if ($args.Count -eq 0) {
    Write-Host "🚀 Running dbt with mock data..." -ForegroundColor Yellow
    dbt run
} else {
    Write-Host "🚀 Running: dbt $args" -ForegroundColor Yellow
    dbt @args
}

Write-Host "✅ Completed with mock data!" -ForegroundColor Green
Write-Host "📊 To run tests: dbt test" -ForegroundColor Cyan
Write-Host "🔄 To disable mock data: `$env:USE_MOCK_DATA='false'" -ForegroundColor Cyan 