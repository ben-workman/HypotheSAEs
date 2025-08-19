import os, time
from openai import OpenAI

model_abbrev_to_id = {
    'gpt4': 'gpt-4-0125-preview',
    'gpt-4': 'gpt-4-0125-preview',
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
}

def get_client():
    api_key = os.environ.get('OPENAI_KEY_SAE') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Set OPENAI_KEY_SAE or OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def _is_responses_model(model_id: str) -> bool:
    m = model_id.lower()
    # Responses for o* and gpt-4.1*; keep gpt-5 on Chat Completions
    return m.startswith("o") or m.startswith("gpt-4.1")

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    client = get_client()
    model_id = model_abbrev_to_id.get(model, model)
    for attempt in range(max_retries):
        try:
            if _is_responses_model(model_id):
                mct = kwargs.pop("max_completion_tokens", None)
                if mct is None and "max_tokens" in kwargs:
                    mct = kwargs.pop("max_tokens")
                resp = client.responses.create(
                    model=model_id,
                    input=[{"role": "user", "content": prompt}],
                    max_completion_tokens=mct,
                    timeout=timeout,
                    **kwargs
                )
                return resp.output_text
            else:
                mt = kwargs.pop("max_tokens", None)
                if mt is None and "max_completion_tokens" in kwargs:
                    mt = kwargs.pop("max_completion_tokens")
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=mt,
                    timeout=timeout,
                    **kwargs
                )
                return resp.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                raise
            wait = timeout * (backoff_factor ** attempt)
            print(f"API timeout, retrying in {wait:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait)
