"""
gradio_app.py — Gradio demo UI for pubmed_rag.

Calls the running FastAPI backend at API_BASE_URL (default: http://localhost:8001).
Run this separately from the API — it is a thin HTTP client, not the pipeline itself.

Usage:
  pip install gradio httpx
  python demo/gradio_app.py                        # http://localhost:7860
  API_BASE_URL=http://100.99.96.88:8001 python demo/gradio_app.py

For screen recording / demo video: open http://localhost:7860 in a browser.
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")
_TIMEOUT = 60  # seconds — pipeline can take 10-15s on CPU

_EXAMPLE_QUERIES = [
    "What is the first-line treatment for HER2-positive metastatic breast cancer?",
    "How does PD-1/PD-L1 checkpoint inhibition restore anti-tumor immunity?",
    "What biomarkers predict response to immune checkpoint inhibitors in solid tumors?",
    "What is the 5-year survival rate for stage III colorectal cancer?",
    "How are PARP inhibitors used in ovarian cancer treatment?",
]

_CSS = """
#answer-box textarea {
    font-size: 15px;
    line-height: 1.6;
}
#sources-box {
    font-size: 14px;
}
.source-row {
    padding: 6px 0;
    border-bottom: 1px solid #e5e7eb;
}
"""


def ask_pubmed(query: str) -> tuple[str, str]:
    """
    Send query to FastAPI /ask, return (answer_text, sources_markdown).

    Formats sources as a numbered markdown list with title, journal, year, and
    a clickable PubMed link — matches the inline [N] citation markers in the answer.
    """
    if not query.strip():
        return "Please enter a clinical question.", ""

    try:
        response = httpx.post(
            f"{_API_BASE}/ask",
            json={"query": query.strip(), "mode": "incremental", "n_results": 5},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        msg = "⚠️ Request timed out. The pipeline may be loading models — try again."
        return msg, ""
    except httpx.HTTPStatusError as e:
        return f"⚠️ API error {e.response.status_code}: {e.response.text[:200]}", ""
    except httpx.ConnectError:
        return f"⚠️ Could not connect to {_API_BASE}. Is the FastAPI server running?", ""

    data = response.json()
    answer = data.get("answer", "No answer returned.")
    sources = data.get("sources", [])
    confidence = data.get("confidence_tier", "")
    coverage_note = data.get("coverage_note")

    # Append confidence + coverage note to the answer
    meta_parts = []
    if confidence:
        emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
        meta_parts.append(f"_{emoji} Retrieval confidence: **{confidence}**_")
    if coverage_note:
        meta_parts.append(f"_⚠️ {coverage_note}_")
    if meta_parts:
        answer += "\n\n" + "  \n".join(meta_parts)

    # Format sources as numbered markdown list
    if not sources:
        sources_md = "_No sources retrieved._"
    else:
        lines = []
        for src in sources:
            pmid = src.get("pmid", "")
            title = src.get("title", "Unknown title")
            journal = src.get("journal", "")
            year = src.get("year", "")
            pubmed_url = src.get("pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            doi_url = src.get("doi_url", "")
            num = src.get("number", "")
            score = src.get("score", 0.0)

            meta = " · ".join(filter(None, [journal, year]))
            links = f"[PubMed ↗]({pubmed_url})"
            if doi_url:
                links += f" · [DOI ↗]({doi_url})"

            lines.append(
                f"**[{num}]** {title}  \n"
                f"&nbsp;&nbsp;&nbsp;{meta} · {links} · _score {score:.2f}_"
            )
        sources_md = "\n\n".join(lines)

    return answer, sources_md


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="PubMed RAG — Oncology Evidence Search") as demo:
        gr.Markdown(
            """
# PubMed RAG — Oncology Evidence Search
**RAG-powered clinical question answering over 35M+ PubMed oncology abstracts.**

Ask any oncology question to get a cited, LLM-synthesised evidence summary.
Each answer cites the specific PubMed abstracts it was generated from.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Clinical question",
                    placeholder=(
                        "e.g. What is the first-line treatment for HER2-positive "
                        "metastatic breast cancer?"
                    ),
                    lines=3,
                    autofocus=True,
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍  Ask PubMed", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                gr.Examples(
                    examples=_EXAMPLE_QUERIES,
                    inputs=query_box,
                    label="Example questions",
                )

        with gr.Column():
            answer_box = gr.Markdown(
                label="Answer",
                value="",
                elem_id="answer-box",
            )

        gr.Markdown("### Sources")
        sources_box = gr.Markdown(
            value="",
            elem_id="sources-box",
        )

        gr.Markdown(
            """
---
**pubmed_rag** · Faithfulness 0.91 (RAGAS) · Recall@20 0.970 (N=97 labeled questions)
[GitHub](https://github.com/omkaradhali/pubmed_rag) · FastAPI + ChromaDB + Anthropic Claude
CDS Hooks endpoint: `/cds-services/pubmed-rag`
            """,
        )

        # Event wiring
        submit_btn.click(
            fn=ask_pubmed,
            inputs=query_box,
            outputs=[answer_box, sources_box],
            show_progress=True,
        )
        query_box.submit(
            fn=ask_pubmed,
            inputs=query_box,
            outputs=[answer_box, sources_box],
            show_progress=True,
        )
        clear_btn.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[query_box, answer_box, sources_box],
        )

    return demo


if __name__ == "__main__":
    print(f"Connecting to API at: {_API_BASE}")
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",   # bind to all interfaces so Tailscale can reach it
        server_port=7860,
        share=False,
        show_error=True,
    )
