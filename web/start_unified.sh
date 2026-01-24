#!/bin/bash
# Ontoloty - Unified Pipeline Web Server Startup Script

echo "=============================================="
echo "  Ontoloty - Unified Multi-Agent Pipeline"
echo "=============================================="

# Navigate to project root
cd "$(dirname "$0")/.."

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Check if unified pipeline modules exist
if [ ! -f "src/unified_pipeline/__init__.py" ]; then
    echo "❌ Unified pipeline not found"
    exit 1
fi

echo "✓ Unified pipeline found"

# Install dependencies if needed
pip install fastapi uvicorn python-multipart --quiet 2>/dev/null

# Start backend
echo ""
echo "Starting Backend API on port 4200..."
echo "  API URL: http://localhost:4200"
echo "  Docs: http://localhost:4200/docs"
echo ""

cd web/backend
python -m uvicorn main_unified:app --host 0.0.0.0 --port 4200 --reload
