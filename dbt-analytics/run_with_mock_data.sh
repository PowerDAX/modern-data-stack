#!/bin/bash

# Script to run dbt with mock data
# Usage: ./run_with_mock_data.sh [dbt_command]

echo "🔧 Setting up mock data environment..."

# Set environment variable
export USE_MOCK_DATA=true

# Seed mock data first
echo "📦 Seeding mock data..."
dbt seed --select mock

# Check if specific dbt command was provided
if [ $# -eq 0 ]; then
    echo "🚀 Running dbt with mock data..."
    dbt run
else
    echo "🚀 Running: dbt $@"
    dbt "$@"
fi

echo "✅ Completed with mock data!"
echo "📊 To run tests: dbt test"
echo "🔄 To disable mock data: export USE_MOCK_DATA=false" 