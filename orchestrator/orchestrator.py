"""
Orchestrator agent — Person 3 (core contribution)

Compares every pair of paper summaries. For each pair:
  1. Comparator agent gives a verdict (agree/contradict/unclear) + confidence
  2. If confidence >= CONFIDENCE_THRESHOLD -> trust it directly
  3. If confidence <  CONFIDENCE_THRESHOLD -> trigger debate:
       - Critic A argues these CONTRADICT
       - Critic B argues these AGREE / don't contradict
       - Judge reads both arguments and makes the final call

Usage:
    python orchestrator/orchestrator.py
    python orchestrator/orchestrator.py --limit 5   # only first 5 papers (fewer pairs)
"""

import argparse
import glob
import itertools
import json
import os
import re
import sys

from groq import Groq
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import PaperSummary, ComparisonResult

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SUMMARIES_DIR = os.path.join(BASE_DIR, "data", "summaries")
DATA_RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

CONFIDENCE_THRESHOLD = 0.85  # below this -> trigger debate
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return raw


def call_llm(prompt: str, max_tokens: int = 400) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def load_summaries(limit: int | None = None) -> list[PaperSummary]:
    files = sorted(glob.glob(os.path.join(DATA_SUMMARIES_DIR, "*.json")))
    if limit:
        files = files[:limit]

    summaries = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        summaries.append(PaperSummary(**data))
    return summaries


def summary_block(s: PaperSummary) -> str:
    return (
        f"Paper: {s.title}\n"
        f"Claim: {s.claim}\n"
        f"Method: {s.method}\n"
        f"Dataset: {s.dataset}\n"
        f"Result: {s.result}"
    )


# ---------------------------------------------------------------------------
# 3a. Comparator agent
# ---------------------------------------------------------------------------

COMPARATOR_PROMPT = """You are comparing two research papers to determine if their
claims and results agree, contradict, or are unclear/unrelated.

{paper_a}

{paper_b}

Respond with ONLY a JSON object with exactly these keys:
- verdict: one of "agree", "contradict", "unclear"
- confidence: a number from 0.0 to 1.0 representing how confident you are in this verdict
- reasoning: 1-2 sentences explaining why

No markdown fences, no extra text, only the JSON object."""


def compare_papers(paper_a: PaperSummary, paper_b: PaperSummary) -> dict | None:
    prompt = COMPARATOR_PROMPT.format(
        paper_a=summary_block(paper_a),
        paper_b=summary_block(paper_b),
    )
    try:
        raw = call_llm(prompt)
        cleaned = clean_json_response(raw)
        parsed = json.loads(cleaned)
        required = {"verdict", "confidence", "reasoning"}
        if not required.issubset(parsed.keys()):
            return None
        return parsed
    except Exception as e:
        print(f"    Comparator failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3c. Debate agents (Critic A, Critic B, Judge)
# ---------------------------------------------------------------------------

CRITIC_A_PROMPT = """You are Critic A. Argue that these two papers CONTRADICT each
other. Be specific about what claim or result conflicts. Keep it to 2-3 sentences.

{paper_a}

{paper_b}

Your argument (that they contradict):"""

CRITIC_B_PROMPT = """You are Critic B. Argue that these two papers DO NOT contradict
each other (they either agree, or are about different enough things that there's no
real conflict). Be specific. Keep it to 2-3 sentences.

{paper_a}

{paper_b}

Your argument (that they do not contradict):"""

JUDGE_PROMPT = """You are the Judge. Two critics have debated whether these two
papers contradict each other. Read both arguments and make a final decision.

{paper_a}

{paper_b}

Critic A (argues they CONTRADICT): {critic_a_argument}

Critic B (argues they DO NOT contradict): {critic_b_argument}

Respond with ONLY a JSON object with exactly these keys:
- verdict: one of "agree", "contradict", "unclear"
- confidence: a number from 0.0 to 1.0
- reasoning: 1-2 sentences explaining your final decision, referencing the stronger argument

No markdown fences, no extra text, only the JSON object."""


def run_debate(paper_a: PaperSummary, paper_b: PaperSummary) -> dict | None:
    pa, pb = summary_block(paper_a), summary_block(paper_b)

    try:
        critic_a_argument = call_llm(
            CRITIC_A_PROMPT.format(paper_a=pa, paper_b=pb), max_tokens=200
        )
        critic_b_argument = call_llm(
            CRITIC_B_PROMPT.format(paper_a=pa, paper_b=pb), max_tokens=200
        )

        judge_raw = call_llm(
            JUDGE_PROMPT.format(
                paper_a=pa,
                paper_b=pb,
                critic_a_argument=critic_a_argument,
                critic_b_argument=critic_b_argument,
            ),
            max_tokens=300,
        )
        cleaned = clean_json_response(judge_raw)
        parsed = json.loads(cleaned)
        required = {"verdict", "confidence", "reasoning"}
        if not required.issubset(parsed.keys()):
            return None
        return parsed

    except Exception as e:
        print(f"    Debate failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3b. Orchestrator (routing logic)
# ---------------------------------------------------------------------------

def orchestrate_pair(paper_a: PaperSummary, paper_b: PaperSummary) -> ComparisonResult | None:
    comparison = compare_papers(paper_a, paper_b)
    if comparison is None:
        return None

    used_debate = False

    if comparison["confidence"] < CONFIDENCE_THRESHOLD:
        print(f"    Confidence {comparison['confidence']:.2f} < {CONFIDENCE_THRESHOLD} -> triggering debate...")
        debate_result = run_debate(paper_a, paper_b)
        if debate_result is not None:
            comparison = debate_result
            used_debate = True
        # if debate fails, we fall back to the original comparator result

    return ComparisonResult(
        paper_id_a=paper_a.paper_id,
        paper_id_b=paper_b.paper_id,
        verdict=comparison["verdict"],
        confidence=float(comparison["confidence"]),
        reasoning=comparison["reasoning"],
        used_debate=used_debate,
    )


def orchestrate_all(limit: int | None = None) -> list[ComparisonResult]:
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

    summaries = load_summaries(limit)
    print(f"Loaded {len(summaries)} paper summaries.")

    pairs = list(itertools.combinations(summaries, 2))
    print(f"Comparing {len(pairs)} pairs...")

    results = []
    for i, (paper_a, paper_b) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {paper_a.title[:40]}... vs {paper_b.title[:40]}...")
        result = orchestrate_pair(paper_a, paper_b)
        if result is None:
            print("    Skipped (comparator failed).")
            continue
        results.append(result)
        print(f"    -> {result.verdict} (confidence {result.confidence:.2f}, debate={result.used_debate})")

    out_path = os.path.join(DATA_RESULTS_DIR, "comparisons.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    print(f"\nDone. Saved {len(results)} comparisons to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all paper pairs and detect contradictions")
    parser.add_argument("--limit", type=int, default=None, help="Only use first N papers")
    args = parser.parse_args()

    orchestrate_all(args.limit)