#!/bin/bash

echo "🚀 Starting Ontology OS (Production Mode)"
echo "=================================================="

# Check Ports (4200: API, 4100: Web)
if lsof -Pi :4200 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 4200 (API) is occupied. Killing..."
    kill -9 $(lsof -Pi :4200 -sTCP:LISTEN -t)
fi
if lsof -Pi :4100 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 4100 (Web) is occupied. Killing..."
    kill -9 $(lsof -Pi :4100 -sTCP:LISTEN -t)
fi

# Start Backend (Port 4200)
echo "📡 Starting Enterprise API (Port 4200)..."
cd web/backend
# --reload used for dev but works for demo
python -m uvicorn main:app --host 0.0.0.0 --port 4200 --reload > api.log 2>&1 &
API_PID=$!
cd ../..

# Start Frontend (Port 4100)
echo "🖥️  Starting Web Interface (Port 4100)..."
cd web/frontend
# Pass port to Next.js
npm run dev -- -p 4100 > web.log 2>&1 &
WEB_PID=$!
cd ../..

echo "=================================================="
echo "✅ System Operational"
echo "👉 Dashboard: http://localhost:4100"
echo "👉 API Server: http://localhost:4200/docs"
echo "=================================================="

# Trap Ctrl+C
trap "kill $API_PID $WEB_PID; exit" INT

wait
