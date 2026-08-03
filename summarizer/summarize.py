"""
Summarizer agent — Person 2

Reads extracted paper text from data/raw/, asks Claude to extract
claim, method, dataset, result as structured JSON, validates it against
schema.PaperSummary, and saves to data/summaries/.

Usage:
    python summarizer/summarize.py
    python summarizer/summarize.py --limit 3     # only process first 3 papers (testing)
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
from schema import PaperSummary

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_SUMMARIES_DIR = os.path.join(BASE_DIR, "data", "summaries")

MAX_CHARS = 15000  # roughly caps how much of each paper we send (keeps cost/time down)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """You are extracting structured facts from a research paper for a
scientific literature review tool. Read the paper text below and extract exactly
these 4 fields:

- claim: the paper's main finding or hypothesis, in 1-2 sentences
- method: how they tested/built it, in 1-2 sentences
- dataset: what data or benchmark they used (name it specifically if given)
- result: the key quantitative or qualitative result, in 1-2 sentences

Respond with ONLY a JSON object with exactly these 4 keys (claim, method, dataset,
result). No markdown fences, no commentary, no extra text before or after the JSON.

Paper title: {title}

Paper text:
{text}
"""


def clean_json_response(raw: str) -> str:
    """Strip markdown fences or stray text around a JSON object."""
    raw = raw.strip()
    # remove ```json ... ``` or ``` ... ``` wrappers
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    # if there's still leading/trailing junk, grab the outermost {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return raw


def summarize_paper(title: str, text: str, retries: int = 2) -> dict | None:
    """Call Claude to extract structured fields. Retries on malformed JSON."""
    truncated_text = text[:MAX_CHARS]
    prompt = PROMPT_TEMPLATE.format(title=title, text=truncated_text)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
)
            raw_text = response.choices[0].message.content
            cleaned = clean_json_response(raw_text)
            parsed = json.loads(cleaned)

            # confirm all 4 required keys are present
            required_keys = {"claim", "method", "dataset", "result"}
            if not required_keys.issubset(parsed.keys()):
                raise ValueError(f"Missing keys, got: {list(parsed.keys())}")

            return parsed

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"    Attempt {attempt + 1} failed to parse: {e}")
            if attempt == retries:
                return None
            continue

    return None


def summarize_all(limit: int | None = None) -> list[PaperSummary]:
    os.makedirs(DATA_SUMMARIES_DIR, exist_ok=True)

    meta_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*.meta.json")))
    if limit:
        meta_files = meta_files[:limit]

    print(f"Found {len(meta_files)} papers to summarize.")

    results = []
    for i, meta_path in enumerate(meta_files, 1):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        paper_id = meta["paper_id"]
        title = meta["title"]
        text_path = meta["raw_text_path"]

        print(f"[{i}/{len(meta_files)}] {title[:70]}...")

        if not os.path.exists(text_path):
            print(f"    Skipping: text file not found at {text_path}")
            continue

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        extracted = summarize_paper(title, text)
        if extracted is None:
            print(f"    Failed after retries, skipping this paper.")
            continue

        summary = PaperSummary(
            paper_id=paper_id,
            title=title,
            claim=extracted["claim"],
            method=extracted["method"],
            dataset=extracted["dataset"],
            result=extracted["result"],
        )

        out_path = os.path.join(DATA_SUMMARIES_DIR, f"{paper_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        results.append(summary)
        print(f"    Saved -> {out_path}")

    print(f"\nDone. Summarized {len(results)}/{len(meta_files)} papers.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize papers into structured JSON")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N papers")
    args = parser.parse_args()

    summarize_all(args.limit)