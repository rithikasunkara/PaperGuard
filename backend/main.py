"""
Backend API — ties Retriever, Summarizer, Orchestrator, and Scorer together.

Usage:
    uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs to test it interactively.
"""

import json
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.fetch import fetch_topic
from summarizer.summarize import summarize_all
from orchestrator.orchestrator import orchestrate_all
from scorer.score import score_all

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

app = FastAPI(title="PaperGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopicRequest(BaseModel):
    topic: str
    max_papers: int = 5


@app.get("/")
def root():
    return {"status": "PaperGuard API is running"}


@app.post("/analyze")
def analyze_topic(request: TopicRequest):
    """
    Full pipeline: fetch -> summarize -> compare -> score.
    Returns summaries, comparisons, and reproducibility scores.
    """
    # 1. Retriever
    papers = fetch_topic(request.topic, request.max_papers)
    if not papers:
        return {"error": "No papers found for this topic."}

    # 2. Summarizer
    summaries = summarize_all(limit=request.max_papers)

    # 3. Orchestrator (comparisons)
    comparisons = orchestrate_all(limit=request.max_papers)

    # 4. Scorer
    scores = score_all(limit=request.max_papers)

    return {
        "topic": request.topic,
        "papers_found": len(papers),
        "summaries": [s.model_dump() for s in summaries],
        "comparisons": [c.model_dump() for c in comparisons],
        "reproducibility_scores": [sc.model_dump() for sc in scores],
    }


@app.get("/results/latest")
def get_latest_results():
    """Read the most recently saved results without re-running the pipeline."""
    comparisons_path = os.path.join(DATA_RESULTS_DIR, "comparisons.json")
    scores_path = os.path.join(DATA_RESULTS_DIR, "reproducibility_scores.json")

    comparisons, scores = [], []

    if os.path.exists(comparisons_path):
        with open(comparisons_path, "r", encoding="utf-8") as f:
            comparisons = json.load(f)

    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)

    return {"comparisons": comparisons, "reproducibility_scores": scores}