import os, time
from openai import OpenAI
from openai import (
    APITimeoutError, APIConnectionError, RateLimitError,
    AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError, APIError
)

model_abbrev_to_id = {
    'gpt4': 'gpt-4-0125-preview',
    'gpt-4': 'gpt-4-0125-preview',
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
}

def get_client():
    api_key = os.environ.get("OPENAI_KEY_SAE") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_KEY_SAE or OPENAI_API_KEY")
    org     = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
    project = os.environ.get("OPENAI_PROJECT_ID") or os.environ.get("OPENAI_PROJECT")
    return OpenAI(api_key=api_key, organization=org, project=project)

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    client  = get_client()
    model_id = model_abbrev_to_id.get(model, model)
    client_t = client.with_options(timeout=timeout)
    is_o_series = model_id.startswith("o")

    for attempt in range(max_retries):
        try:
            if is_o_series:
                kwargs_resp = dict(kwargs)
                if "max_tokens" in kwargs_resp:
                    kwargs_resp["max_output_tokens"] = kwargs_resp.pop("max_tokens")
                kwargs_resp.pop("temperature", None)
                resp = client_t.responses.create(model=model_id, input=prompt, **kwargs_resp)
                return resp.output_text
            else:
                resp = client_t.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return resp.choices[0].message.content

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            wait = timeout * (backoff_factor ** attempt)
            print(f"{type(e).__name__}: retrying in {wait:.1f}s... ({attempt+1}/{max_retries})")
            time.sleep(wait)

        except (AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError, APIError) as e:
            print(f"OpenAI {type(e).__name__}: {e}")
            raise
