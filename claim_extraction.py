import ollama
import json


def extract_claims(text, model='llama3'):
    prompt = f"""Break the following text into a list of atomic factual claims. Each claim should be a single, standalone, verifiable statement. Return ONLY a JSON array of strings, no other text, no markdown formatting.

Text: {text}

JSON array:"""

    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    raw = response['message']['content'].strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        claims = json.loads(raw[start:end])

    return claims
