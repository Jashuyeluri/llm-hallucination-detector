import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tokenizer = None
_model = None
_label2id = None
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli"


def _load():
    global _tokenizer, _model, _label2id
    if _model is None:
        # This model uses a SentencePiece tokenizer.  Keeping the slow
        # tokenizer prevents newer Transformers releases from attempting to
        # convert its binary spm.model file as if it were a TikToken model.
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        _label2id = {v.lower(): k for k, v in _model.config.id2label.items()}
    return _tokenizer, _model, _label2id


def check_claim(source_text, claim):
    tokenizer, model, label2id = _load()

    inputs = tokenizer.encode(
        source_text, claim,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():
        logits = model(inputs)[0]

    probs = torch.softmax(logits, dim=1)[0]
    contra_score = probs[label2id["contradiction"]].item()
    neutral_score = probs[label2id["neutral"]].item()
    entail_score = probs[label2id["entailment"]].item()

    if entail_score > contra_score and entail_score > neutral_score:
        label = "supported"
    elif contra_score > entail_score and contra_score > neutral_score:
        label = "contradicted"
    else:
        label = "unsupported"

    return {
        "claim": claim,
        "label": label,
        "entailment_score": round(entail_score, 4),
        "contradiction_score": round(contra_score, 4),
        "neutral_score": round(neutral_score, 4),
    }


def check_all_claims(source_text, claims):
    return [check_claim(source_text, c) for c in claims]
