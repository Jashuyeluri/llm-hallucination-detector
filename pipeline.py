from claim_extraction import extract_claims
from entailment_check import check_all_claims
from scoring import compute_faithfulness_score
from correction import correct_response


def run_pipeline(source_text, llm_response, model=None, auto_correct=True):
    claims = extract_claims(llm_response, model=model)
    results = check_all_claims(source_text, claims, model=model)
    score = compute_faithfulness_score(results)

    corrected_response = None
    if auto_correct and (score["contradicted"] > 0 or score["unsupported"] > 0):
        corrected_response = correct_response(source_text, llm_response, results, model=model)

    return {
        "claims": claims,
        "results": results,
        "score": score,
        "corrected_response": corrected_response,
    }
