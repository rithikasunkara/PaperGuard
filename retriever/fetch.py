"""
Retriever agent — Person 1

Fetches papers on a given topic from arXiv, downloads the PDFs,
extracts clean text with PyMuPDF, and saves everything to data/raw/.

Usage:
    python retriever/fetch.py "contradiction detection in NLP" --max_results 10
"""

import argparse
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
import requests

# schema.py lives one folder up
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import PaperMeta

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def search_arxiv(topic: str, max_results: int = 10) -> list[dict]:
    """Query arXiv's API and return a list of paper metadata dicts."""
    query = urllib.parse.quote(f"all:{topic}")
    url = f"{ARXIV_API_URL}?search_query={query}&start=0&max_results={max_results}"

    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")

    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        published = entry.find("atom:published", ns).text[:4]  # year only

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        papers.append({
            "paper_id": arxiv_id,
            "title": title,
            "authors": authors,
            "year": int(published) if published.isdigit() else None,
            "source_url": pdf_url,
        })

    return papers


def download_pdf(pdf_url: str, save_path: str) -> None:
    """Download a PDF from a URL to a local path."""
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts)
    # collapse excessive whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text


def safe_filename(paper_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", paper_id)


def fetch_topic(topic: str, max_results: int = 10) -> list[PaperMeta]:
    """Full pipeline: search -> download -> extract -> save. Returns list of PaperMeta."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    print(f"Searching arXiv for: '{topic}' (max {max_results} results)...")
    papers = search_arxiv(topic, max_results)
    print(f"Found {len(papers)} papers.")

    results = []
    for i, paper in enumerate(papers, 1):
        fname = safe_filename(paper["paper_id"])
        pdf_path = os.path.join(DATA_RAW_DIR, f"{fname}.pdf")
        text_path = os.path.join(DATA_RAW_DIR, f"{fname}.txt")
        meta_path = os.path.join(DATA_RAW_DIR, f"{fname}.meta.json")

        print(f"[{i}/{len(papers)}] {paper['title'][:70]}...")

        try:
            download_pdf(paper["source_url"], pdf_path)
            text = extract_text(pdf_path)

            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            meta = PaperMeta(
                paper_id=paper["paper_id"],
                title=paper["title"],
                authors=paper["authors"],
                year=paper["year"],
                source_url=paper["source_url"],
                raw_text_path=text_path,
            )

            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(meta.model_dump_json(indent=2))

            results.append(meta)
            time.sleep(1)  # be polite to arXiv's servers

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print(f"\nDone. Saved {len(results)} papers to {DATA_RAW_DIR}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv on a topic")
    parser.add_argument("topic", type=str, help="Search topic, e.g. 'graph neural networks'")
    parser.add_argument("--max_results", type=int, default=10, help="Number of papers to fetch")
    args = parser.parse_args()

    fetch_topic(args.topic, args.max_results)