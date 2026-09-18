import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Groq production model ID. It can be changed without a code deployment by
# setting GROQ_MODEL in the host's environment variables.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def chat(prompt, model=None):
    """Sends a single-turn prompt to whichever LLM provider is configured
    and returns the response text. Set LLM_PROVIDER=groq (with GROQ_API_KEY)
    for cloud/hosted deployments, or leave as 'ollama' for local use."""
    if LLM_PROVIDER == "groq":
        return _chat_groq(prompt, model or GROQ_MODEL)
    if LLM_PROVIDER == "ollama":
        return _chat_ollama(prompt, model or OLLAMA_MODEL)
    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'groq'.")


def _chat_ollama(prompt, model):
    import ollama
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()


def _chat_groq(prompt, model):
    from groq import Groq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()
