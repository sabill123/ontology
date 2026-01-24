#!/bin/bash

echo "🚀 Starting Ontology OS..."
echo "=================================================="

# Check if port 8000 is occupied
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is occupied. Killing process..."
    kill -9 $(lsof -Pi :8000 -sTCP:LISTEN -t)
fi

# Check if port 8501 is occupied
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8501 is occupied. Killing process..."
    kill -9 $(lsof -Pi :8501 -sTCP:LISTEN -t)
fi

# Start Backend API in background
echo "📡 Starting Backend API (Port 8000)..."
python -m uvicorn src.app.main_service:app --host 0.0.0.0 --port 8000 --workers 1 > api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 3

# Start Frontend UI
echo "🖥️  Starting Frontend UI (Port 8501)..."
streamlit run src/app/ui.py > ui.log 2>&1 &
UI_PID=$!

echo "=================================================="
echo "✅ Ontology OS is running!"
echo "👉 Dashboard: http://localhost:8501"
echo "👉 API Docs:  http://localhost:8000/docs"
echo "=================================================="
echo "Press Ctrl+C to stop all services."

# Trap Ctrl+C
trap "kill $API_PID $UI_PID; exit" INT

# Keep script running
wait
