"""Standalone runner for the PDF demo pipeline with OpenAI GPT integration."""

import os
import sys
from typing import Optional

from openai import OpenAI

from main import run_demo


def _build_client() -> OpenAI:
    """Create an OpenAI client using environment variables."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variável de ambiente OPENAI_API_KEY ausente. "
            "Defina sua chave antes de rodar o demo."
        )

    organization = os.getenv("OPENAI_ORGANIZATION")
    return OpenAI(api_key=api_key, organization=organization)


CLIENT = _build_client()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def llm_complete(system_prompt: str, user_prompt: str, *, model: Optional[str] = None) -> str:
    """Call OpenAI's Chat Completions API."""

    target_model = model or MODEL_NAME
    completion = CLIENT.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "2014-IDEAS-Vania.pdf"
    resultado = run_demo(pdf_path, llm_complete)

    print("Chunks gerados:", resultado.chunk_count)
    print("Resumo final:\n", resultado.final_summary)
    print("Resposta padrão:\n", resultado.answer)
