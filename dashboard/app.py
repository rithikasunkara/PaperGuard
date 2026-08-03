"""
Streamlit dashboard — Person 4b

The visual front-end for PaperGuard. Lets you type a topic, trigger the full
pipeline via the backend API, and view agreements/contradictions/reproducibility
scores.

Usage:
    streamlit run dashboard/app.py

Make sure the backend is running first:
    uvicorn backend.main:app --reload
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="PaperGuard", page_icon="📄", layout="wide")

st.title("📄 PaperGuard")
st.caption("Adaptive multi-agent contradiction detection and reproducibility scoring for research papers")

# ---------------------------------------------------------------------------
# Sidebar: run a new analysis
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Run a new analysis")
    topic = st.text_input("Research topic", placeholder="e.g. transformer attention efficiency")
    max_papers = st.slider("Number of papers", min_value=2, max_value=10, value=5)
    run_button = st.button("Analyze", type="primary")

    st.divider()
    st.caption(
        "This calls the full pipeline (Retriever → Summarizer → Orchestrator → "
        "Scorer) and can take a few minutes depending on paper count."
    )
    load_existing = st.button("Load existing results (fast)")


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

def run_full_analysis(topic: str, max_papers: int):
    with st.spinner(f"Analyzing '{topic}'... this can take a few minutes."):
        response = requests.post(
            f"{API_BASE}/analyze",
            json={"topic": topic, "max_papers": max_papers},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()


def load_latest_results():
    response = requests.get(f"{API_BASE}/results/latest", timeout=30)
    response.raise_for_status()
    return response.json()


data = None

if run_button:
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        try:
            data = run_full_analysis(topic, max_papers)
        except Exception as e:
            st.error(f"Analysis failed: {e}")

elif load_existing:
    try:
        data = load_latest_results()
    except Exception as e:
        st.error(f"Could not load existing results: {e}")


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if data:
    comparisons = data.get("comparisons", [])
    scores = data.get("reproducibility_scores", [])

    tab1, tab2, tab3 = st.tabs(["🔍 Agreements & Contradictions", "✅ Reproducibility Scores", "📊 Overview"])

    # --- Tab 1: comparisons ---
    with tab1:
        if not comparisons:
            st.info("No comparison results yet. Run an analysis with at least 2 papers.")
        else:
            verdict_colors = {"agree": "green", "contradict": "red", "unclear": "orange"}

            for c in comparisons:
                verdict = c.get("verdict", "unclear")
                color = verdict_colors.get(verdict, "gray")
                debate_tag = " 🗣️ *(resolved via debate)*" if c.get("used_debate") else ""

                with st.container(border=True):
                    st.markdown(f"**Paper A:** `{c['paper_id_a']}` &nbsp;&nbsp; **Paper B:** `{c['paper_id_b']}`")
                    st.markdown(
                        f":{color}[**{verdict.upper()}**] "
                        f"(confidence: {c['confidence']:.2f}){debate_tag}"
                    )
                    st.caption(c.get("reasoning", ""))

    # --- Tab 2: reproducibility scores ---
    with tab2:
        if not scores:
            st.info("No reproducibility scores yet.")
        else:
            for s in scores:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{s['paper_id']}**")
                        st.caption(s.get("notes", ""))
                    with col2:
                        st.metric("Score", f"{s['score']}/100")

                    checklist = s.get("checklist", {})
                    cols = st.columns(len(checklist)) if checklist else []
                    for col, (item, passed) in zip(cols, checklist.items()):
                        with col:
                            icon = "✅" if passed else "❌"
                            st.caption(f"{icon} {item.replace('_', ' ')}")

    # --- Tab 3: overview ---
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Comparisons made", len(comparisons))
        with col2:
            contradictions = sum(1 for c in comparisons if c.get("verdict") == "contradict")
            st.metric("Contradictions found", contradictions)
        with col3:
            avg_score = (
                round(sum(s["score"] for s in scores) / len(scores), 1) if scores else 0
            )
            st.metric("Avg. reproducibility score", f"{avg_score}/100")

        debates = sum(1 for c in comparisons if c.get("used_debate"))
        st.caption(f"{debates} of {len(comparisons)} comparisons required the debate step (low initial confidence).")

else:
    st.info("Enter a topic in the sidebar and click **Analyze**, or click **Load existing results** to view your last run.")