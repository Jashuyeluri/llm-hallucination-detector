"""Source-grounded claim verification using the configured LLM provider.

This replaces the local PyTorch/DeBERTa model so the application can run on a
small cloud instance. The verifier is still source-only: it must not use its
own background knowledge when assigning a label.
"""

import json

from llm_client import chat


VALID_LABELS = {"supported", "contradicted", "unsupported"}


def _json_from_response(raw):
    """Extract JSON from an LLM response that may include a Markdown fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        raise ValueError("Verifier did not return a JSON array.")
    return json.loads(text[start:end])


def _result(claim, label, confidence):
    """Keep the result schema used by scoring.py and the browser UI."""
    confidence = max(0.0, min(float(confidence), 1.0))
    if label == "supported":
        entailment, contradiction, neutral = confidence, 0.0, 1.0 - confidence
    elif label == "contradicted":
        entailment, contradiction, neutral = 0.0, confidence, 1.0 - confidence
    else:
        entailment, contradiction, neutral = 0.0, 0.0, confidence

    return {
        "claim": claim,
        "label": label,
        "entailment_score": round(entailment, 4),
        "contradiction_score": round(contradiction, 4),
        "neutral_score": round(neutral, 4),
    }


def check_all_claims(source_text, claims, model=None):
    """Classify all claims in one source-grounded LLM request.

    Batching reduces API calls, latency, and cost. A claim is supported only
    when the supplied source supports it; missing evidence is unsupported.
    """
    if not claims:
        return []

    numbered_claims = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(claims)
    )
    prompt = f"""You are a strict source-grounded fact verifier.

Use ONLY the source material below. Do not use outside knowledge and do not
infer facts that the source does not state.

SOURCE MATERIAL:
{source_text}

CLAIMS TO VERIFY:
{numbered_claims}

For every claim, return exactly one JSON array. Each array item must contain:
- index: the claim number
- label: one of supported, contradicted, unsupported
- confidence: a number from 0 to 1

Rules:
- supported: the source explicitly supports the claim.
- contradicted: the source explicitly states an incompatible fact.
- unsupported: the source does not provide enough evidence either way.

Return only JSON. Do not add Markdown, explanations, or extra keys."""

    parsed = _json_from_response(chat(prompt, model=model))
    by_index = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        label = str(item.get("label", "")).lower().strip()
        if isinstance(index, int) and 0 <= index < len(claims) and label in VALID_LABELS:
            by_index[index] = _result(claims[index], label, item.get("confidence", 0.5))

    # An incomplete or malformed item is deliberately treated as unsupported,
    # never as a verified fact.
    return [
        by_index.get(index, _result(claim, "unsupported", 0.0))
        for index, claim in enumerate(claims)
    ]
