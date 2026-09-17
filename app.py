"""
EcoIntel AI - FastAPI Application

AI Biodiversity Assessment & Decision Support System

Endpoints:
    POST /analyze     - Run environmental analysis (returns report or follow-up questions)
    POST /assessment  - Run full assessment (returns structured report)
    GET  /health      - Service health check
"""

import logging
import os
import json
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from modules.models import AnalysisRequest
from modules.workflow import EcoIntelWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the EcoIntel workflow on startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("EcoIntel AI - Starting up...")
    logger.info("=" * 60)

    try:
        app.state.workflow = EcoIntelWorkflow(
            data_dir=os.getenv("ECOINTEL_DATA_DIR", "data"),
            vectordb_dir=os.getenv("ECOINTEL_VECTORDB_DIR", "vectordb"),
        )
        logger.info("EcoIntel AI initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize EcoIntel AI: {e}", exc_info=True)
        raise

    yield

    logger.info("EcoIntel AI - Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EcoIntel AI",
    description=(
        "AI Biodiversity Assessment & Decision Support System. "
        "Analyzes ecosystem conditions, identifies biodiversity risks, "
        "and generates evidence-backed recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory if it exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the interactive Web Dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "EcoIntel AI API running. Access /health or /docs"}


@app.get("/health")
async def health_check():
    """Service health check with knowledge base statistics."""
    try:
        kb_stats = app.state.workflow.get_knowledge_base_stats()
    except Exception:
        kb_stats = {"status": "unavailable"}

    return {
        "status": "healthy",
        "service": "EcoIntel AI",
        "version": "1.0.0",
        "description": "AI Biodiversity Assessment & Decision Support System",
        "knowledge_base_stats": kb_stats,
    }


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """
    Analyze environmental data and return results.

    Accepts either natural language input or structured environmental data.
    Returns either a completed assessment report or follow-up questions
    if critical data is missing.
    """
    workflow: EcoIntelWorkflow = app.state.workflow
    raw_input = _extract_raw_input(request)

    try:
        result = workflow.run(raw_input)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        if result.get("assessment_report"):
            return {
                "status": "completed",
                "result": result["assessment_report"],
            }
        elif result.get("follow_up_questions"):
            return {
                "status": "needs_more_info",
                "message": "Additional environmental data is needed for a complete assessment.",
                "questions": result["follow_up_questions"],
                "missing_fields": result.get("missing_fields", []),
            }
        else:
            return {"status": "incomplete", "state": _safe_serialize(result)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assessment")
async def assessment(request: AnalysisRequest):
    """
    Run a full environmental assessment and return a structured report.

    Same input as /analyze but guarantees a structured AssessmentReport
    in the response (or follow-up questions if data is insufficient).
    """
    workflow: EcoIntelWorkflow = app.state.workflow
    raw_input = _extract_raw_input(request)

    try:
        result = workflow.run(raw_input)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        if result.get("assessment_report"):
            report = result["assessment_report"]
            return {
                "status": "completed",
                "assessment_report": report,
            }
        elif result.get("follow_up_questions"):
            return {
                "status": "needs_more_info",
                "message": "Additional environmental data is needed for a complete assessment.",
                "questions": result["follow_up_questions"],
                "missing_fields": result.get("missing_fields", []),
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Workflow did not produce a report or follow-up questions.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_raw_input(request: AnalysisRequest) -> str:
    """Extract raw input string from the request."""
    if request.natural_language:
        return request.natural_language
    elif request.input_data:
        return request.input_data.model_dump_json(exclude_none=True)
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'natural_language' or 'input_data'.",
        )


def _safe_serialize(data: dict) -> dict:
    """Safely serialize state for JSON response, handling non-serializable types."""
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        return {k: str(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
