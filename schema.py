"""
Shared data schema for PaperGuard.
Every agent (Retriever, Summarizer, Orchestrator, Scorer) reads/writes
these exact shapes so the pipeline connects cleanly.
"""

from pydantic import BaseModel
from typing import Optional


class PaperMeta(BaseModel):
    """Output of the Retriever agent."""
    paper_id: str          # e.g. arXiv id or Semantic Scholar id
    title: str
    authors: list[str]
    year: Optional[int] = None
    source_url: str
    raw_text_path: str     # where the extracted PDF text is saved


class PaperSummary(BaseModel):
    """Output of the Summarizer agent. This is the core shared object."""
    paper_id: str
    title: str
    claim: str              # main finding/hypothesis
    method: str              # how they tested it
    dataset: str             # what data they used
    result: str               # what they found


class ComparisonResult(BaseModel):
    """Output of the Comparator agent (inside Orchestrator)."""
    paper_id_a: str
    paper_id_b: str
    verdict: str              # "agree" | "contradict" | "unclear"
    confidence: float         # 0.0 - 1.0
    reasoning: str
    used_debate: bool = False


class ReproducibilityScore(BaseModel):
    """Output of the Reproducibility Scorer agent."""
    paper_id: str
    score: float               # 0-100
    checklist: dict[str, bool]  # e.g. {"dataset_named": True, "code_available": False}
    notes: str
