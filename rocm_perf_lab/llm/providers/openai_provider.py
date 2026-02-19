import os
from typing import Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def openai_llm_callable(model: str = "gpt-4.1", temperature: float = 0.2) -> Callable[[str], str]:
    """
    Returns a callable that sends prompts to OpenAI Chat Completions API.
    Requires OPENAI_API_KEY in environment.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")

    client = OpenAI(api_key=api_key)

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are an expert AMD GPU performance engineer."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    return _call
