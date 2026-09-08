# LLM Hallucination Detector — Verification Lab dashboard

A custom-designed web dashboard (FastAPI backend + hand-built HTML/CSS/JS frontend) replacing the earlier Streamlit UI, styled as a fact-verification report rather than a generic AI dashboard.

## Structure

```
claim_extraction.py    stage 2 — Llama 3 extracts atomic claims
entailment_check.py    stage 3 — DeBERTa-v3-MNLI checks each claim against the source
scoring.py              stage 4 — faithfulness score aggregation
correction.py           stage 5 — Llama 3 rewrites the response using verified facts
pipeline.py             orchestrates source-document verification
file_utils.py           reads uploaded .txt/.pdf files
server.py               FastAPI backend, exposes /api/check and serves the frontend
static/
  index.html            dashboard markup
  style.css             design system (paper background, serif/mono type, evidence-card UI)
  app.js                mode switching, file uploads, API calls, results rendering
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3
```

## Run

```bash
uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000** in your browser — this serves the custom dashboard directly (not Streamlit).

## Notes

- First run downloads `MoritzLaurer/DeBERTa-v3-base-mnli` (~370MB) automatically.
- Ollama must be running locally (`ollama serve`) before using the app.
- The frontend calls the backend at the same origin (`/api/check`), so no separate frontend server is needed — `uvicorn` serves both the API and the static dashboard.
