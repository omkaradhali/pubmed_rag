"""
gradio_app.py — Gradio demo UI for pubmed_rag.

Calls the running FastAPI backend at API_BASE_URL (default: http://localhost:8001).

Usage:
  pip install gradio httpx
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
    "What biomarkers predict response to immune checkpoint inhibitors?",
    "What is the 5-year survival rate for stage III colorectal cancer?",
    "How are PARP inhibitors used in ovarian cancer treatment?",
]

_CSS = """
/* ── Global ─────────────────────────────────────────────── */
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* ── Banner ──────────────────────────────────────────────── */
.pubmed-banner {
    background: linear-gradient(135deg, #0f4c81 0%, #1565c0 60%, #1a73b8 100%);
    border-radius: 12px;
    padding: 26px 30px 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(15, 76, 129, 0.2);
}
.pubmed-banner h1 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
}
.pubmed-banner p {
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.9rem;
    margin: 0 0 14px 0;
    line-height: 1.5;
}
.banner-badges { display: flex; flex-wrap: wrap; gap: 6px; }
.b-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* ── Answer card — force all text dark regardless of theme ── */
#answer-panel,
#answer-panel *,
#answer-panel p,
#answer-panel li,
#answer-panel h1,
#answer-panel h2,
#answer-panel h3,
#answer-panel span {
    color: #1f2937 !important;
}
#answer-panel strong, #answer-panel b { color: #111827 !important; }

/* ── Query input ─────────────────────────────────────────── */
#query-input textarea {
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
    transition: border-color 0.15s !important;
}
#query-input textarea::placeholder { color: #9ca3af !important; }
#query-input textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    outline: none !important;
}

/* ── Suggested question chips ────────────────────────────── */
.chips-row {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    padding: 2px 0 !important;
}
.q-chip button {
    background: #eff6ff !important;
    color: #2563eb !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: unset !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
    transition: background 0.12s, border-color 0.12s !important;
    box-shadow: none !important;
}
.q-chip button:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
#ask-btn button {
    background: #2563eb !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    height: 44px !important;
    font-size: 0.92rem !important;
}
#ask-btn button:hover { background: #1d4ed8 !important; }
#clear-btn button {
    border-radius: 8px !important;
    height: 44px !important;
    border: 1.5px solid #d1d5db !important;
    background: white !important;
    color: #6b7280 !important;
    font-size: 0.88rem !important;
}

/* ── Answer card ─────────────────────────────────────────── */
#answer-panel {
    border-left: 4px solid #2563eb !important;
    border-radius: 0 8px 8px 0 !important;
    background: #f8faff !important;
    padding: 16px 20px !important;
    min-height: 80px !important;
    font-size: 0.97rem !important;
    line-height: 1.78 !important;
}

/* ── Confidence row ──────────────────────────────────────── */
.conf-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
}
.conf-high   { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.conf-medium { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
.conf-low    { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.conf-unknown{ background: #f9fafb; color: #6b7280; border: 1px solid #e5e7eb; }
.coverage-note {
    margin-top: 8px;
    padding: 8px 12px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #92400e;
}

/* ── Source cards ────────────────────────────────────────── */
.ref-card {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 13px 15px;
    margin-bottom: 8px;
    transition: border-color 0.12s;
}
.ref-card:hover { border-color: #93c5fd; }
.ref-num {
    min-width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #2563eb;
    color: white;
    font-size: 0.73rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.ref-body { flex: 1; }
.ref-title {
    font-weight: 600;
    font-size: 0.89rem;
    color: #111827;
    line-height: 1.4;
    margin-bottom: 4px;
}
.ref-meta { font-size: 0.78rem; color: #9ca3af; margin-bottom: 7px; }
.ref-link {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 0.73rem;
    font-weight: 600;
    text-decoration: none;
    margin-right: 5px;
}
.ref-link:hover { background: #dbeafe; }
.ref-score {
    font-size: 0.72rem;
    color: #9ca3af;
    float: right;
    margin-top: 2px;
}

/* ── Empty / error states ────────────────────────────────── */
.placeholder {
    text-align: center;
    color: #9ca3af;
    font-size: 0.86rem;
    padding: 20px 0;
    line-height: 1.6;
}
.err-box {
    padding: 12px 14px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 7px;
    color: #991b1b;
    font-size: 0.86rem;
}

/* ── Footer ──────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    font-size: 0.74rem;
    color: #9ca3af;
    margin-top: 6px;
    padding-top: 14px;
    border-top: 1px solid #f3f4f6;
}
.app-footer a { color: #9ca3af; text-decoration: none; }
.app-footer a:hover { color: #6b7280; }
"""

_SEC_STYLE = (
    "font-size:0.71rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.09em;color:#374151;margin:0 0 10px 0;"
    "padding-bottom:8px;border-bottom:1.5px solid #d1d5db;"
)

_BANNER = """
<div class="pubmed-banner">
  <h1>🔬 PubMed RAG &mdash; Oncology Evidence Search</h1>
  <p>RAG-powered clinical question answering over 4,992 curated PubMed oncology abstracts.
  Every answer is grounded in retrieved literature with inline citations.</p>
  <div class="banner-badges">
    <span class="b-badge">📄 4,992 abstracts</span>
    <span class="b-badge">✅ Faithfulness 0.91</span>
    <span class="b-badge">🎯 Recall@20 0.97</span>
    <span class="b-badge">🔗 HL7 CDS Hooks</span>
  </div>
</div>
"""

_FOOTER = """
<div class="app-footer">
  <a href="https://github.com/omkaradhali/pubmed_rag" target="_blank">GitHub</a> ·
  FastAPI + ChromaDB + Anthropic Claude ·
  CDS Hooks: <code style="background:#f3f4f6;border:1px solid #e5e7eb;
  border-radius:4px;padding:1px 5px;font-size:0.7rem">/cds-services/pubmed-rag</code>
</div>
"""

_PH_STYLE = (
    "text-align:center;color:#9ca3af;font-size:0.86rem;"
    "padding:20px 0;line-height:1.6;"
)
_ANSWER_PLACEHOLDER = (
    f"<div style='{_PH_STYLE}'>Ask a question above to see the evidence summary here.</div>"
)
_SOURCES_PLACEHOLDER = (
    f"<div style='{_PH_STYLE}'>Retrieved PubMed abstracts will appear here "
    "with title, journal, year, and links.</div>"
)


# ── Result builders ───────────────────────────────────────────────────────────


def _build_confidence(tier: str, coverage_note: str | None) -> str:
    icons = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    colors = {
        "High":   ("bg:#f0fdf4", "#16a34a", "#bbf7d0"),
        "Medium": ("bg:#fffbeb", "#d97706", "#fde68a"),
        "Low":    ("bg:#fef2f2", "#dc2626", "#fecaca"),
    }.get(tier, ("bg:#f9fafb", "#6b7280", "#e5e7eb"))
    bg, fg, border = colors[0].split(":")[1], colors[1], colors[2]
    icon = icons.get(tier, "⚪")
    style = (
        f"display:inline-flex;align-items:center;gap:6px;"
        f"padding:5px 12px;border-radius:20px;font-size:0.78rem;"
        f"font-weight:600;margin-top:10px;background:{bg};"
        f"color:{fg};border:1px solid {border};"
    )
    pill = f'<div><span style="{style}">{icon} Retrieval confidence: {tier}</span>'
    if coverage_note:
        warn = (
            "margin-top:8px;padding:8px 12px;background:#fffbeb;"
            "border:1px solid #fde68a;border-radius:6px;"
            "font-size:0.8rem;color:#92400e;"
        )
        pill += f'<div style="{warn}">⚠️ {coverage_note}</div>'
    return pill + "</div>"


def _build_sources(sources: list[dict]) -> str:
    if not sources:
        return "<div class='placeholder'>No sources retrieved.</div>"

    n = len(sources)
    sec = (
        "font-size:0.71rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.09em;color:#374151;margin:0 0 10px 0;"
        "padding-bottom:8px;border-bottom:1.5px solid #d1d5db;"
    )
    header = f'<p style="{sec}">{n} Reference{"s" if n != 1 else ""} Retrieved</p>'
    cards = []
    for src in sources:
        num = src.get("number", "?")
        title = src.get("title", "Untitled")
        journal = src.get("journal", "")
        year = src.get("year", "")
        pmid = src.get("pmid", "")
        pubmed_url = src.get("pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        doi_url = src.get("doi_url", "")
        score = src.get("score", 0.0)

        meta_parts = [p for p in [journal, year, f"PMID {pmid}" if pmid else ""] if p]
        meta = " · ".join(meta_parts)
        doi_link = (
            f'<a class="ref-link" href="{doi_url}" target="_blank"'
            ' style="color:#2563eb">DOI ↗</a>'
            if doi_url else ""
        )
        cards.append(f"""
<div class="ref-card">
  <div class="ref-num">{num}</div>
  <div class="ref-body">
    <div class="ref-title" style="color:#111827;font-weight:600;font-size:0.89rem;
         line-height:1.4;margin-bottom:4px">{title}</div>
    <div class="ref-meta" style="color:#6b7280;font-size:0.78rem;margin-bottom:7px"
         >{meta}</div>
    <span class="ref-score" style="color:#9ca3af;font-size:0.72rem;float:right;
          margin-top:2px">score {score:.2f}</span>
    <a class="ref-link" href="{pubmed_url}" target="_blank"
       style="color:#2563eb">PubMed ↗</a>
    {doi_link}
  </div>
</div>""")

    return header + "\n".join(cards)


# ── Query handler ─────────────────────────────────────────────────────────────


def ask_pubmed(query: str) -> tuple[str, str, str]:
    """Return (answer_markdown, confidence_html, sources_html)."""
    if not query.strip():
        return "", _ANSWER_PLACEHOLDER, _SOURCES_PLACEHOLDER

    try:
        resp = httpx.post(
            f"{_API_BASE}/ask",
            json={"query": query.strip(), "mode": "incremental", "n_results": 5},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        err = "<div class='err-box'>⏱ Request timed out — try again in a few seconds.</div>"
        return "", err, _SOURCES_PLACEHOLDER
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        msg = e.response.text[:200]
        err = f"<div class='err-box'>⚠️ API error {code}: {msg}</div>"
        return "", err, _SOURCES_PLACEHOLDER
    except httpx.ConnectError:
        err = (
            f"<div class='err-box'>🔌 Cannot connect to {_API_BASE} "
            "— is the FastAPI server running?</div>"
        )
        return "", err, _SOURCES_PLACEHOLDER

    data = resp.json()
    answer = data.get("answer", "No answer returned.")
    sources = data.get("sources", [])
    confidence = data.get("confidence_tier", "")
    coverage_note = data.get("coverage_note")

    return (
        answer,
        _build_confidence(confidence, coverage_note),
        _build_sources(sources),
    )


# ── Layout ────────────────────────────────────────────────────────────────────


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="PubMed RAG — Oncology Evidence Search") as demo:

        gr.HTML(_BANNER)

        # ── Ask a question ─────────────────────────────────────
        with gr.Group():
            gr.HTML(f'<p style="{_SEC_STYLE}">Clinical Question</p>')
            query_box = gr.Textbox(
                label="",
                show_label=False,
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
                    "Clear",
                    scale=1,
                    elem_id="clear-btn",
                )

        with gr.Group():
            gr.HTML(f'<p style="{_SEC_STYLE}">Suggested Questions</p>')
            with gr.Row(elem_classes="chips-row"):
                chips = [
                    gr.Button(q, elem_classes="q-chip", size="sm")
                    for q in _EXAMPLE_QUERIES
                ]

        # ── Evidence summary ───────────────────────────────────
        with gr.Group():
            gr.HTML(f'<p style="{_SEC_STYLE}">Evidence Summary</p>')
            answer_box = gr.Markdown(
                value=_ANSWER_PLACEHOLDER,
                elem_id="answer-panel",
            )
            confidence_box = gr.HTML(value="")

        # ── References ─────────────────────────────────────────
        with gr.Group():
            sources_box = gr.HTML(value=_SOURCES_PLACEHOLDER)

        gr.HTML(_FOOTER)

        # ── Wiring ─────────────────────────────────────────────
        outputs = [answer_box, confidence_box, sources_box]

        submit_btn.click(
            fn=ask_pubmed,
            inputs=query_box,
            outputs=outputs,
            show_progress="full",
        )
        query_box.submit(
            fn=ask_pubmed,
            inputs=query_box,
            outputs=outputs,
            show_progress="full",
        )
        clear_btn.click(
            fn=lambda: ("", _ANSWER_PLACEHOLDER, "", _SOURCES_PLACEHOLDER),
            inputs=[],
            outputs=[query_box] + outputs,
        )

        # Each chip populates the query box with its label text
        for chip in chips:
            chip.click(
                fn=lambda q=chip.value: q,
                inputs=[],
                outputs=[query_box],
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
