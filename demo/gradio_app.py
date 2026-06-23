"""
gradio_app.py — Gradio demo UI for pubmed_rag.

Calls the running FastAPI backend at API_BASE_URL (default: http://localhost:8001).
Run this separately from the API — it is a thin HTTP client, not the pipeline itself.

Usage:
  pip install gradio httpx
  python demo/gradio_app.py                              # http://localhost:7860
  API_BASE_URL=http://100.99.96.88:8011 python demo/gradio_app.py
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")
_TIMEOUT = 60

_EXAMPLE_QUERIES = [
    "What is the first-line treatment for HER2-positive metastatic breast cancer?",
    "How does PD-1/PD-L1 checkpoint inhibition restore anti-tumor immunity?",
    "What biomarkers predict response to immune checkpoint inhibitors in solid tumors?",
    "What is the 5-year survival rate for stage III colorectal cancer?",
    "How are PARP inhibitors used in ovarian cancer treatment?",
]

# ── CSS design system ─────────────────────────────────────────────────────────

_CSS = """
/* ── Global ────────────────────────────────────────────────── */
.gradio-container {
    max-width: 880px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: #f3f4f6 !important;
    padding-bottom: 32px !important;
}
body, .dark { background: #f3f4f6 !important; }

/* ── Header ─────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f4c81 0%, #1565c0 60%, #1a73b8 100%);
    border-radius: 14px;
    padding: 30px 34px 24px;
    margin-bottom: 6px;
    box-shadow: 0 4px 20px rgba(15, 76, 129, 0.25);
}
.app-header h1 {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.app-header .subtitle {
    font-size: 0.93rem;
    color: rgba(255,255,255,0.88);
    line-height: 1.55;
    margin-bottom: 16px;
}
.badge-row { display: flex; flex-wrap: wrap; gap: 7px; }
.hdr-badge {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    border-radius: 20px;
    padding: 4px 13px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

/* ── Section cards ──────────────────────────────────────────── */
.section-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 22px 24px;
    margin-top: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.section-label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #9ca3af !important;
    margin-bottom: 14px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #f3f4f6;
}

/* ── Query input ────────────────────────────────────────────── */
#query-input textarea {
    font-size: 0.97rem !important;
    line-height: 1.6 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    background: #fafafa !important;
    color: #111827 !important;
    resize: vertical !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
#query-input textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    background: white !important;
    outline: none !important;
}
#query-input label span {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
#ask-btn button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    border: none !important;
    height: 46px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    transition: all 0.15s !important;
}
#ask-btn button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}
#clear-btn button {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    height: 46px !important;
    color: #6b7280 !important;
    border: 1.5px solid #e5e7eb !important;
    background: white !important;
    transition: all 0.15s !important;
}
#clear-btn button:hover {
    border-color: #d1d5db !important;
    color: #374151 !important;
    background: #f9fafb !important;
}

/* ── Answer panel ───────────────────────────────────────────── */
#answer-panel {
    border-left: 4px solid #2563eb !important;
    border-radius: 0 8px 8px 0 !important;
    background: #f8faff !important;
    padding: 18px 22px !important;
    margin: 0 !important;
}
#answer-panel .prose p,
#answer-panel p {
    font-size: 0.97rem !important;
    line-height: 1.8 !important;
    color: #1f2937 !important;
    margin-bottom: 10px !important;
}
#answer-panel .prose strong,
#answer-panel strong { color: #111827 !important; }

/* ── Source cards ───────────────────────────────────────────── */
.src-cards-wrap { display: flex; flex-direction: column; gap: 10px; }
.src-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    gap: 13px;
    align-items: flex-start;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.src-card:hover {
    border-color: #93c5fd;
    box-shadow: 0 3px 10px rgba(37,99,235,0.09);
}
.src-num {
    background: #2563eb;
    color: white;
    border-radius: 50%;
    min-width: 28px;
    height: 28px;
    font-size: 0.74rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
    box-shadow: 0 2px 4px rgba(37,99,235,0.25);
}
.src-body { flex: 1; min-width: 0; }
.src-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.45;
    margin-bottom: 5px;
}
.src-meta {
    font-size: 0.79rem;
    color: #9ca3af;
    margin-bottom: 9px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.src-meta-sep { color: #d1d5db; }
.src-links { display: flex; gap: 7px; flex-wrap: wrap; align-items: center; }
.src-link {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    text-decoration: none;
    transition: background 0.12s;
    cursor: pointer;
}
.src-link:hover { background: #dbeafe; }
.src-score {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6b7280;
    margin-left: auto;
    white-space: nowrap;
    padding: 2px 8px;
    background: #f9fafb;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
}

/* ── Confidence row ─────────────────────────────────────────── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #f9fafb;
    border: 1px solid #f3f4f6;
    border-radius: 8px;
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 12px;
}
.conf-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.conf-high  { background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,0.2); }
.conf-medium{ background: #f59e0b; box-shadow: 0 0 0 3px rgba(245,158,11,0.2); }
.conf-low   { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.2); }
.conf-none  { background: #9ca3af; }
.conf-label { font-weight: 600; color: #374151; }
.coverage-warn {
    margin-top: 6px;
    padding: 8px 12px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 7px;
    font-size: 0.8rem;
    color: #92400e;
}

/* ── Empty / error states ───────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 32px 20px;
    color: #9ca3af;
    font-size: 0.88rem;
    line-height: 1.6;
}
.empty-icon { font-size: 2rem; margin-bottom: 10px; }
.error-state {
    padding: 14px 16px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    color: #991b1b;
    font-size: 0.87rem;
    font-weight: 500;
}

/* ── Footer ─────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    font-size: 0.76rem;
    color: #9ca3af;
    padding: 20px 0 6px;
    margin-top: 8px;
}
.app-footer a { color: #9ca3af; text-decoration: none; }
.app-footer a:hover { color: #6b7280; }
.app-footer code {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.72rem;
    color: #374151;
}
"""

# ── Static HTML fragments ─────────────────────────────────────────────────────

_HEADER = """
<div class="app-header">
  <h1>🔬 PubMed RAG — Oncology Evidence Search</h1>
  <div class="subtitle">
    RAG-powered clinical question answering over a curated corpus of PubMed oncology
    abstracts. Every answer is grounded in retrieved literature — no hallucinated sources.
  </div>
  <div class="badge-row">
    <span class="hdr-badge">📄 4,992 oncology abstracts</span>
    <span class="hdr-badge">✅ Faithfulness 0.91 (RAGAS)</span>
    <span class="hdr-badge">🎯 Recall@20 0.97</span>
    <span class="hdr-badge">🔗 HL7 CDS Hooks compatible</span>
  </div>
</div>
"""

_ANSWER_EMPTY = """
<div class="empty-state">
  <div class="empty-icon">💬</div>
  Ask a clinical question above to see a cited evidence summary here.
</div>
"""

_SOURCES_EMPTY = """
<div class="empty-state">
  <div class="empty-icon">📚</div>
  Retrieved PubMed sources will appear here, each with title, journal, year,
  relevance score, and a direct link to the abstract.
</div>
"""

_FOOTER = """
<div class="app-footer">
  <a href="https://github.com/omkaradhali/pubmed_rag" target="_blank">GitHub</a>
  &nbsp;·&nbsp; FastAPI + ChromaDB + Anthropic Claude
  &nbsp;·&nbsp; CDS Hooks: <code>/cds-services/pubmed-rag</code>
</div>
"""


# ── Data helpers ──────────────────────────────────────────────────────────────


def _confidence_html(tier: str, coverage_note: str | None) -> str:
    dot_class = {
        "High": "conf-high",
        "Medium": "conf-medium",
        "Low": "conf-low",
    }.get(tier, "conf-none")
    label = tier or "Unknown"
    warn = (
        f'<div class="coverage-warn">⚠️ {coverage_note}</div>'
        if coverage_note
        else ""
    )
    return f"""
<div>
  <div class="conf-row">
    <span class="conf-dot {dot_class}"></span>
    <span>Retrieval confidence: <span class="conf-label">{label}</span></span>
  </div>
  {warn}
</div>
"""


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return (
            '<div class="empty-state"><div class="empty-icon">🔍</div>'
            "No sources retrieved.</div>"
        )

    count = len(sources)
    cards = []
    for src in sources:
        num = src.get("number", "?")
        title = src.get("title", "Unknown title")
        journal = src.get("journal", "")
        year = src.get("year", "")
        pmid = src.get("pmid", "")
        pubmed_url = src.get("pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        doi_url = src.get("doi_url", "")
        score = src.get("score", 0.0)

        meta_parts = []
        if journal:
            meta_parts.append(f'<span>{journal}</span>')
        if year:
            sep = '<span class="src-meta-sep">·</span>' if meta_parts else ""
            meta_parts.append(f'{sep}<span>{year}</span>')
        if pmid:
            sep = '<span class="src-meta-sep">·</span>' if meta_parts else ""
            meta_parts.append(f'{sep}<span>PMID {pmid}</span>')
        meta_html = " ".join(meta_parts)

        doi_link = (
            f'<a class="src-link" href="{doi_url}" target="_blank">DOI ↗</a>'
            if doi_url else ""
        )

        cards.append(f"""
<div class="src-card">
  <div class="src-num">{num}</div>
  <div class="src-body">
    <div class="src-title">{title}</div>
    <div class="src-meta">{meta_html}</div>
    <div class="src-links">
      <a class="src-link" href="{pubmed_url}" target="_blank">PubMed ↗</a>
      {doi_link}
      <span class="src-score">score {score:.2f}</span>
    </div>
  </div>
</div>""")

    cards_html = "\n".join(cards)
    return f"""
<div>
  <div class="section-label">{count} Reference{"s" if count != 1 else ""} Retrieved</div>
  <div class="src-cards-wrap">{cards_html}</div>
</div>
"""


# ── Main query function ───────────────────────────────────────────────────────


def ask_pubmed(query: str) -> tuple[str, str, str]:
    """Return (answer_markdown, confidence_html, sources_html)."""
    if not query.strip():
        return "", _ANSWER_EMPTY, _SOURCES_EMPTY

    try:
        response = httpx.post(
            f"{_API_BASE}/ask",
            json={"query": query.strip(), "mode": "incremental", "n_results": 5},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        err = (
            '<div class="error-state">⏱ Request timed out — the pipeline may be '
            "loading models. Try again in a few seconds.</div>"
        )
        return "", err, _SOURCES_EMPTY
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        detail = e.response.text[:180]
        err = f'<div class="error-state">⚠️ API error {code}: {detail}</div>'
        return "", err, _SOURCES_EMPTY
    except httpx.ConnectError:
        err = (
            f'<div class="error-state">🔌 Could not connect to {_API_BASE}. '
            "Is the FastAPI server running?</div>"
        )
        return "", err, _SOURCES_EMPTY

    data = response.json()
    answer = data.get("answer", "No answer returned.")
    sources = data.get("sources", [])
    confidence = data.get("confidence_tier", "")
    coverage_note = data.get("coverage_note")

    return (
        answer,
        _confidence_html(confidence, coverage_note),
        _sources_html(sources),
    )


# ── UI layout ─────────────────────────────────────────────────────────────────


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="PubMed RAG — Oncology Evidence Search") as demo:

        # Header
        gr.HTML(_HEADER)

        # ── Query section ──────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML('<div class="section-label">Clinical Question</div>')
            query_box = gr.Textbox(
                label="",
                placeholder=(
                    "e.g. What is the first-line treatment for HER2-positive "
                    "metastatic breast cancer?"
                ),
                lines=3,
                autofocus=True,
                elem_id="query-input",
            )
            with gr.Row():
                submit_btn = gr.Button(
                    "🔍  Search PubMed",
                    variant="primary",
                    scale=3,
                    elem_id="ask-btn",
                )
                clear_btn = gr.Button(
                    "✕  Clear",
                    scale=1,
                    elem_id="clear-btn",
                )

        gr.Examples(
            examples=_EXAMPLE_QUERIES,
            inputs=query_box,
            label="Example clinical questions",
        )

        # ── Answer section ─────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            gr.HTML('<div class="section-label">Evidence Summary</div>')
            answer_box = gr.Markdown(
                value=_ANSWER_EMPTY,
                elem_id="answer-panel",
            )
            confidence_box = gr.HTML(value="")

        # ── Sources section ────────────────────────────────────
        with gr.Group(elem_classes="section-card"):
            sources_box = gr.HTML(value=_SOURCES_EMPTY)

        # Footer
        gr.HTML(_FOOTER)

        # ── Event wiring ───────────────────────────────────────
        outputs = [answer_box, confidence_box, sources_box]

        submit_btn.click(
            fn=ask_pubmed,
            inputs=query_box,
            outputs=outputs,
            show_progress="minimal",
        )
        query_box.submit(
            fn=ask_pubmed,
            inputs=query_box,
            outputs=outputs,
            show_progress="minimal",
        )
        clear_btn.click(
            fn=lambda: ("", "", _ANSWER_EMPTY, _SOURCES_EMPTY),
            inputs=[],
            outputs=[query_box] + outputs,
        )

    return demo


if __name__ == "__main__":
    print(f"Connecting to API at: {_API_BASE}")
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=_CSS,
    )
