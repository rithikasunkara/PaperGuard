# PaperGuard

An adaptive multi-agent system for contradiction detection and reproducibility
scoring in research papers.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
cp .env.example .env         # then fill in your API keys
```

## Folder structure

```
retriever/      Person 1 — fetch papers (arXiv, Semantic Scholar), extract text
summarizer/      Person 2 — LLM prompts to extract claim/method/dataset/result
orchestrator/      Person 3 — Comparator, confidence routing, debate agents
scorer/          Person 4a — reproducibility checklist scoring
backend/          FastAPI app tying agents together
dashboard/        Streamlit UI
data/raw/          extracted paper text (gitignored)
data/summaries/      structured JSON summaries (gitignored)
data/results/          final comparison + score outputs (gitignored)
schema.py          shared Pydantic models — READ THIS FIRST
```

## Shared contract

Every agent reads/writes the exact shapes defined in `schema.py`:
`PaperMeta` → `PaperSummary` → `ComparisonResult` / `ReproducibilityScore`.

Do not invent your own field names — if the schema needs a new field, add it
to `schema.py` and tell the team, so nobody's code breaks silently.

## Run order (local dev)

1. `retriever/fetch.py` — pulls papers on a topic into `data/raw/`
2. `summarizer/summarize.py` — turns raw text into `data/summaries/*.json`
3. `orchestrator/orchestrator.py` — compares summaries, writes `data/results/`
4. `scorer/score.py` — scores reproducibility, appends to `data/results/`
5. `uvicorn backend.main:app --reload` — start API
6. `streamlit run dashboard/app.py` — start dashboard
