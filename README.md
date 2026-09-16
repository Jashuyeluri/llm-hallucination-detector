# LLM Hallucination Detector — Verification Lab dashboard

A custom-designed web dashboard (FastAPI backend + hand-built HTML/CSS/JS frontend) replacing the earlier Streamlit UI, styled as a fact-verification report rather than a generic AI dashboard.

## Structure

```
claim_extraction.py    stage 2 — configured LLM extracts atomic claims
entailment_check.py    stage 3 — configured LLM checks each claim against the source
scoring.py              stage 4 — faithfulness score aggregation
correction.py           stage 5 — configured LLM rewrites the response using verified facts
llm_client.py            provider adapter for Ollama or Groq
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

- By default, the app uses Ollama, so Ollama must be running locally (`ollama serve`).
- The frontend calls the backend at the same origin (`/api/check`), so no separate frontend server is needed — `uvicorn` serves both the API and the static dashboard.

## Deploy to Render with Groq

The Docker image is ready for Groq. It uses Groq for claim extraction,
source-grounded verification, and response correction; it does not download a
local PyTorch or DeBERTa model. In Render, create a **Web Service** from this
repository, choose **Docker**, and set the health-check path to `/api/health`.

Add these environment variables in Render's dashboard:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=your_private_groq_key
GROQ_MODEL=openai/gpt-oss-20b
```

Do not store `GROQ_API_KEY` in the repository or Dockerfile. `GROQ_MODEL` may
be changed to another Groq model ID without modifying the application code.
