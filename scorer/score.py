"""
Reproducibility Scorer agent — Person 4a

Reads each paper's summary (and raw text for more detail), checks the method
against a reproducibility checklist using the LLM, scores it 0-100, and saves
to data/results/.

Usage:
    python scorer/score.py
    python scorer/score.py --limit 3
"""

import argparse
import glob
import json
import os
import re
import sys

from groq import Groq
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import PaperSummary, ReproducibilityScore

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_SUMMARIES_DIR = os.path.join(BASE_DIR, "data", "summaries")
DATA_RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

MAX_CHARS = 15000
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CHECKLIST_ITEMS = [
    "dataset_named",
    "sample_size_given",
    "hyperparameters_given",
    "code_availability",
    "data_availability",
    "evaluation_metric_defined",
    "baseline_comparison",
    "hardware_or_compute_stated",
]

SCORER_PROMPT = """You are assessing the reproducibility of a research paper's method
section. For each checklist item below, answer true or false based on whether the
paper's text explicitly provides that information.

Checklist items:
1. dataset_named - is the dataset/benchmark specifically named (not just "a dataset")?
2. sample_size_given - is the sample size / data volume stated?
3. hyperparameters_given - are key hyperparameters (learning rate, batch size, epochs, etc.) stated?
4. code_availability - is code/repository availability mentioned?
5. data_availability - is data availability/access mentioned?
6. evaluation_metric_defined - are the evaluation metrics clearly defined?
7. baseline_comparison - is there a baseline/comparison method described?
8. hardware_or_compute_stated - is hardware/compute environment mentioned?

Paper title: {title}

Paper text:
{text}

Respond with ONLY a JSON object with exactly these keys:
- checklist: an object with all 8 items above as keys, each with a true/false value
- notes: 1-2 sentences summarizing the paper's reproducibility strengths/weaknesses

No markdown fences, no extra text, only the JSON object."""


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return raw


def score_paper(title: str, text: str, retries: int = 2) -> dict | None:
    truncated_text = text[:MAX_CHARS]
    prompt = SCORER_PROMPT.format(title=title, text=truncated_text)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content
            cleaned = clean_json_response(raw_text)
            parsed = json.loads(cleaned)

            if "checklist" not in parsed or "notes" not in parsed:
                raise ValueError(f"Missing keys, got: {list(parsed.keys())}")

            missing_items = set(CHECKLIST_ITEMS) - set(parsed["checklist"].keys())
            if missing_items:
                raise ValueError(f"Missing checklist items: {missing_items}")

            return parsed

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"    Attempt {attempt + 1} failed to parse: {e}")
            if attempt == retries:
                return None
            continue

    return None


def compute_score(checklist: dict) -> float:
    """Each checklist item is worth an equal share of 100 points."""
    true_count = sum(1 for v in checklist.values() if v is True)
    return round((true_count / len(CHECKLIST_ITEMS)) * 100, 1)


def score_all(limit: int | None = None) -> list[ReproducibilityScore]:
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

    summary_files = sorted(glob.glob(os.path.join(DATA_SUMMARIES_DIR, "*.json")))
    if limit:
        summary_files = summary_files[:limit]

    print(f"Found {len(summary_files)} papers to score.")

    results = []
    for i, summary_path in enumerate(summary_files, 1):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        summary = PaperSummary(**summary_data)
        paper_id = summary.paper_id
        title = summary.title

        print(f"[{i}/{len(summary_files)}] {title[:70]}...")

        raw_text_path = os.path.join(DATA_RAW_DIR, f"{paper_id}.txt")
        if os.path.exists(raw_text_path):
            with open(raw_text_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = f"Method: {summary.method}\nDataset: {summary.dataset}\nResult: {summary.result}"

        scored = score_paper(title, text)
        if scored is None:
            print(f"    Failed after retries, skipping this paper.")
            continue

        repro_score = ReproducibilityScore(
            paper_id=paper_id,
            score=compute_score(scored["checklist"]),
            checklist=scored["checklist"],
            notes=scored["notes"],
        )

        results.append(repro_score)
        print(f"    Score: {repro_score.score}/100")

    out_path = os.path.join(DATA_RESULTS_DIR, "reproducibility_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    print(f"\nDone. Scored {len(results)}/{len(summary_files)} papers. Saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score papers for reproducibility")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N papers")
    args = parser.parse_args()

    score_all(args.limit)