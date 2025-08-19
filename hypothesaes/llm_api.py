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

CHAT_PASS = {'temperature','top_p','n','stop','presence_penalty','frequency_penalty','logit_bias','user','seed','store','metadata'}
RESP_PASS = {'reasoning_effort','metadata','seed','store','tool_choice','parallel_tool_calls'}

def get_client():
    k = os.environ.get('OPENAI_KEY_SAE') or os.environ.get('OPENAI_API_KEY')
    if not k: raise ValueError("Set OPENAI_KEY_SAE or OPENAI_API_KEY")
    return OpenAI(api_key=k)

def _is_responses_model(m: str) -> bool:
    ml = m.lower()
    return ml.startswith(("o", "gpt-4.1"))

def get_completion(prompt: str, model: str = "gpt-4o", timeout: float = 15.0, max_retries: int = 3, backoff_factor: float = 2.0, **kwargs) -> str:
    client = get_client()
    model_id = model_abbrev_to_id.get(model, model)
    for attempt in range(max_retries):
        try:
            kw = dict(kwargs)
            if model_id.lower().startswith("gpt-5"):
                kw.pop("temperature", None)
                kw.pop("reasoning_effort", None)
                kw.pop("max_tokens", None)
                kw.pop("max_output_tokens", None)
                kw.pop("max_completion_tokens", None)
                ckw = {k:v for k,v in kw.items() if k in CHAT_PASS}
                resp = client.chat.completions.create(model=model_id, messages=[{"role":"user","content":prompt}], timeout=timeout, **ckw)
                return resp.choices[0].message.content

            if _is_responses_model(model_id):
                kw.pop("temperature", None)
                mct = kw.pop("max_output_tokens", None) or kw.pop("max_completion_tokens", None) or kw.pop("max_tokens", None)
                rkw = {k:v for k,v in kw.items() if k in RESP_PASS}
                resp = client.responses.create(model=model_id, input=prompt, max_output_tokens=mct, timeout=timeout, **rkw)
                return resp.output_text

            mt = kw.pop("max_tokens", None) or kw.pop("max_output_tokens", None) or kw.pop("max_completion_tokens", None)
            kw.pop("reasoning_effort", None)
            ckw = {k:v for k,v in kw.items() if k in CHAT_PASS}
            resp = client.chat.completions.create(model=model_id, messages=[{"role":"user","content":prompt}], max_tokens=mt, timeout=timeout, **ckw)
            return resp.choices[0].message.content

        except Exception:
            if attempt == max_retries - 1:
                raise
            wait = timeout * (backoff_factor ** attempt)
            print(f"API timeout, retrying in {wait:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait)
