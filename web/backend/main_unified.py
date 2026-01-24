"""
Ontoloty Backend - Unified Pipeline Integration
Clean integration with the new unified multi-agent pipeline
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root to sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# --- Unified Pipeline Import ---
from src.unified_pipeline import UnifiedPipelineOrchestrator, SharedContext
from src.common.utils.logger import logger

app = FastAPI(title="Ontoloty API", version="3.0.0", description="Unified Multi-Agent Ontology Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4100", "http://localhost:3000", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# System State
# ============================================================================
class SystemState:
    def __init__(self):
        self.active_scenario: str = "fintech_banking"
        self.logs: List[str] = []
        self.pipeline_status: str = "idle"  # idle, running, completed, error
        self.pipeline_progress: Dict[str, Any] = {}
        self.pipeline_results: Optional[Dict[str, Any]] = None
        self.orchestrator: Optional[UnifiedPipelineOrchestrator] = None
        self.websocket_clients: List[WebSocket] = []

    def add_log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        self.logs.append(formatted)
        if len(self.logs) > 200:
            self.logs.pop(0)
        # Broadcast to websocket clients (only if event loop is running)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast_log(formatted))
        except RuntimeError:
            pass  # No event loop running, skip broadcast

    async def broadcast_log(self, log: str):
        for ws in self.websocket_clients[:]:
            try:
                await ws.send_json({"type": "log", "data": log})
            except:
                self.websocket_clients.remove(ws)

state = SystemState()
state.add_log("Unified Pipeline Backend initialized")


# ============================================================================
# API Models
# ============================================================================
class ScenarioRequest(BaseModel):
    scenario: str
    domain: Optional[str] = None

class PipelineStatus(BaseModel):
    status: str
    progress: Dict[str, Any]
    current_stage: Optional[str] = None

class OntologyResponse(BaseModel):
    concepts: List[Dict[str, Any]]
    statistics: Dict[str, int]

class GovernanceResponse(BaseModel):
    decisions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "version": "3.0.0",
        "active_scenario": state.active_scenario,
        "pipeline_status": state.pipeline_status,
        "pipeline_progress": state.pipeline_progress,
        "logs": state.logs[-50:]
    }


@app.get("/api/scenarios")
async def list_scenarios():
    """List available scenarios"""
    scenarios_dir = ROOT_DIR / "data" / "scenarios"
    scenarios = []

    if scenarios_dir.exists():
        for path in scenarios_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Check if it has data files
                csv_count = len(list(path.glob("*.csv")))
                has_output = (path / "unified_output").exists()
                scenarios.append({
                    "name": path.name,
                    "tables": csv_count,
                    "has_results": has_output
                })

    return {"scenarios": scenarios}


@app.post("/api/scenario/select")
async def select_scenario(request: ScenarioRequest):
    """Select active scenario"""
    state.active_scenario = request.scenario
    state.add_log(f"Scenario changed to: {request.scenario}")
    return {"status": "ok", "active_scenario": state.active_scenario}


# ============================================================================
# Phase 1: Data Sources
# ============================================================================

@app.get("/api/phase1/sources")
async def get_sources():
    """Get data sources for current scenario"""
    scenario_dir = ROOT_DIR / "data" / "scenarios" / state.active_scenario
    sources = []

    if scenario_dir.exists():
        for csv_file in scenario_dir.glob("*.csv"):
            import csv as csv_module
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv_module.reader(f)
                headers = next(reader, [])
                row_count = sum(1 for _ in reader)

            sources.append({
                "name": csv_file.stem,
                "type": "csv",
                "table_count": 1,
                "row_count": row_count,
                "column_count": len(headers),
                "columns": headers[:10],
                "quality_score": 0.85
            })

    return sources


@app.get("/api/phase1/schema")
async def get_schema():
    """Get schema info for current scenario"""
    scenario_dir = ROOT_DIR / "data" / "scenarios" / state.active_scenario
    tables = []

    if scenario_dir.exists():
        for csv_file in scenario_dir.glob("*.csv"):
            import csv as csv_module
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv_module.reader(f)
                headers = next(reader, [])
                row_count = sum(1 for _ in reader)

            tables.append({
                "name": csv_file.stem,
                "row_count": row_count,
                "column_count": len(headers),
                "columns": headers,
                "primary_keys": [h for h in headers if h.endswith("_id") or h == "id"],
                "foreign_keys": [h for h in headers if h.endswith("_id") and h != "id"]
            })

    return {"tables": tables}


@app.get("/api/phase1/files")
async def get_files():
    """Get uploaded files"""
    scenario_dir = ROOT_DIR / "data" / "scenarios" / state.active_scenario
    files = []

    if scenario_dir.exists():
        for f in scenario_dir.glob("*.csv"):
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })

    return files


@app.post("/api/phase1/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files to scenario"""
    upload_dir = ROOT_DIR / "data" / "scenarios" / state.active_scenario
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for file in files:
        file_path = upload_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        uploaded.append(file.filename)
        state.add_log(f"Uploaded: {file.filename}")

    return {"status": "ok", "files": uploaded}


@app.get("/api/phase1/concepts")
async def get_concepts():
    """Get generated concepts"""
    output_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "ontology.json"

    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {"concepts": data.get("concepts", []), "statistics": data.get("statistics", {})}

    return {"concepts": [], "statistics": {}}


@app.get("/api/phase1/output")
async def get_phase1_output():
    """Get Phase 1 output (discovery results)"""
    output_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "unified_pipeline_context.json"

    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract discovery stage results
        discovery = data.get("stage_results", {}).get("discovery", {})
        return {
            "homeomorphisms": discovery.get("homeomorphisms", []),
            "unified_entities": discovery.get("unified_entities", []),
            "value_overlaps": discovery.get("value_overlaps", []),
            "domain_context": data.get("domain_context", state.active_scenario.split("_")[0]),
            "topology_score": discovery.get("summary", {}).get("avg_confidence", 0.75)
        }

    return {"error": "No output found. Run pipeline first."}


# ============================================================================
# Phase 2: Insights
# ============================================================================

@app.get("/api/phase2/insights")
async def get_insights():
    """Get generated insights"""
    output_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "ontology.json"

    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        insights = []
        for concept in data.get("concepts", []):
            insights.append({
                "id": concept.get("id"),
                "explanation": concept.get("description", ""),
                "confidence": concept.get("confidence", 0.8),
                "severity": "high" if concept.get("confidence", 0) > 0.8 else "medium",
                "category": concept.get("type")
            })

        return insights

    return []


@app.get("/api/phase2/output")
async def get_phase2_output():
    """Get Phase 2 output (refinement results)"""
    output_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "unified_pipeline_context.json"

    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        refinement = data.get("stage_results", {}).get("refinement", {})
        ontology = data.get("ontology", {})

        return {
            "concepts": ontology.get("concepts", []),
            "conflicts_resolved": refinement.get("summary", {}).get("conflicts_resolved", 0),
            "validated_count": refinement.get("summary", {}).get("concepts_generated", 0),
            "approved": ontology.get("statistics", {}).get("approved", 0),
            "provisional": ontology.get("statistics", {}).get("provisional", 0)
        }

    return {"error": "No output found. Run pipeline first."}


# ============================================================================
# Phase 3: Governance
# ============================================================================

@app.get("/api/phase3/output")
async def get_phase3_output():
    """Get Phase 3 output (governance results)"""
    governance_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "governance_decisions.json"
    actions_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "action_backlog.json"

    result = {}

    if governance_file.exists():
        with open(governance_file, 'r', encoding='utf-8') as f:
            result["governance"] = json.load(f)

    if actions_file.exists():
        with open(actions_file, 'r', encoding='utf-8') as f:
            result["actions"] = json.load(f)

    if result:
        return result

    return {"error": "No output found. Run pipeline first."}


@app.get("/api/phase3/decisions")
async def get_decisions():
    """Get governance decisions"""
    governance_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "governance_decisions.json"

    if governance_file.exists():
        with open(governance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("decisions", [])

    return []


@app.get("/api/phase3/actions")
async def get_actions():
    """Get action backlog"""
    actions_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "action_backlog.json"

    if actions_file.exists():
        with open(actions_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return []


# ============================================================================
# Pipeline Execution
# ============================================================================

@app.post("/api/pipeline/run")
async def run_pipeline(background_tasks: BackgroundTasks, request: Optional[ScenarioRequest] = None):
    """Run unified pipeline"""
    if state.pipeline_status == "running":
        raise HTTPException(status_code=400, detail="Pipeline already running")

    scenario = request.scenario if request else state.active_scenario
    domain = request.domain if request else scenario.split("_")[0]

    state.pipeline_status = "running"
    state.pipeline_progress = {"stage": "initializing", "progress": 0}
    state.add_log(f"Starting unified pipeline for scenario: {scenario}")

    background_tasks.add_task(execute_pipeline, scenario, domain)

    return {"status": "started", "scenario": scenario}


async def execute_pipeline(scenario: str, domain: str):
    """Execute unified pipeline in background"""
    try:
        scenario_dir = ROOT_DIR / "data" / "scenarios" / scenario
        output_dir = scenario_dir / "unified_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        state.add_log(f"Initializing pipeline for {scenario}")

        # Create orchestrator
        orchestrator = UnifiedPipelineOrchestrator(
            scenario_name=scenario,
            domain_context=domain,
            use_llm=True,
            output_dir=output_dir
        )
        state.orchestrator = orchestrator

        # Load tables
        state.add_log("Loading tables...")
        tables_data = orchestrator.load_tables(scenario_dir)
        state.add_log(f"Loaded {len(tables_data)} tables")

        # Run pipeline with streaming
        event_count = 0
        for event in orchestrator.run(tables_data):
            event_count += 1
            event_type = event.get("type", "info")

            if event_type == "stage_start":
                stage = event.get("stage", "unknown")
                state.pipeline_progress = {"stage": stage, "progress": 0}
                state.add_log(f"Stage started: {stage}")

            elif event_type == "stage_complete":
                stage = event.get("stage", "unknown")
                state.pipeline_progress = {"stage": stage, "progress": 100}
                state.add_log(f"Stage completed: {stage}")

            elif event_type == "task_start":
                task = event.get("task", "unknown")
                state.add_log(f"  → {task}")

            elif event_type == "task_complete":
                task = event.get("task", "unknown")
                result = event.get("result", "")
                if result:
                    state.add_log(f"  ✓ {task}: {result}")
                else:
                    state.add_log(f"  ✓ {task}")

            elif event_type == "pipeline_complete":
                state.add_log("Pipeline completed successfully")

        # Save results
        state.add_log("Saving results...")
        orchestrator.save_results(output_dir)

        state.pipeline_status = "completed"
        state.pipeline_progress = {"stage": "completed", "progress": 100}
        state.add_log(f"Total events processed: {event_count}")

        # Load results into state
        summary_file = output_dir / "pipeline_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                state.pipeline_results = json.load(f)

    except Exception as e:
        state.pipeline_status = "error"
        state.add_log(f"Pipeline error: {str(e)}")
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get pipeline execution status"""
    return {
        "status": state.pipeline_status,
        "progress": state.pipeline_progress,
        "results": state.pipeline_results
    }


@app.post("/api/phase1/run")
async def run_phase1(background_tasks: BackgroundTasks):
    """Run Phase 1 (Discovery) only"""
    # For compatibility - runs full pipeline but user sees Phase 1 results first
    return await run_pipeline(background_tasks)


@app.post("/api/phase2/run")
async def run_phase2(background_tasks: BackgroundTasks):
    """Run Phase 2 (Refinement)"""
    # Pipeline runs all stages, frontend shows phase 2 results
    if state.pipeline_status != "completed":
        return await run_pipeline(background_tasks)
    return {"status": "already_completed", "message": "Use existing results"}


@app.post("/api/phase3/run")
async def run_phase3(background_tasks: BackgroundTasks):
    """Run Phase 3 (Governance)"""
    # Pipeline runs all stages, frontend shows phase 3 results
    if state.pipeline_status != "completed":
        return await run_pipeline(background_tasks)
    return {"status": "already_completed", "message": "Use existing results"}


@app.post("/api/run_pipeline")
async def run_pipeline_legacy(background_tasks: BackgroundTasks):
    """Legacy endpoint - runs full pipeline"""
    return await run_pipeline(background_tasks)


# ============================================================================
# WebSocket for Real-time Logs
# ============================================================================

@app.websocket("/api/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time logs"""
    await websocket.accept()
    state.websocket_clients.append(websocket)

    try:
        # Send existing logs
        await websocket.send_json({"type": "init", "logs": state.logs})

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text("ping")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.websocket_clients:
            state.websocket_clients.remove(websocket)


# ============================================================================
# Results Endpoints
# ============================================================================

@app.get("/api/results/summary")
async def get_results_summary():
    """Get pipeline results summary"""
    summary_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "pipeline_summary.json"

    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {"error": "No results found"}


@app.get("/api/results/ontology")
async def get_ontology():
    """Get generated ontology"""
    ontology_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "ontology.json"

    if ontology_file.exists():
        with open(ontology_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {"error": "No ontology found"}


@app.get("/api/results/conversations")
async def get_conversations():
    """Get agent conversations"""
    conv_file = ROOT_DIR / "data" / "scenarios" / state.active_scenario / "unified_output" / "agent_conversations.json"

    if conv_file.exists():
        with open(conv_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {"error": "No conversations found"}


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4200)
