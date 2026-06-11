"""LLM memo drafter — Azure OpenAI (Microsoft Foundry) backend for the
review agent's memo_drafter hook.

Configuration via environment (never in code, never in Git):
    AZURE_OPENAI_API_KEY      the deployment key
    AZURE_OPENAI_ENDPOINT     e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_DEPLOYMENT   deployment name (default: gpt-4.1-mini)

Design principles:
- Rules decide, models write prose. The LLM receives the facts (the
  rules-engine explanation and the template memo as a fact sheet) and is
  instructed to rewrite, never to add, soften, or re-decide anything.
- Graceful degradation, as everywhere else in this system: if the env
  vars are absent, the package isn't installed, or the call fails, the
  agent falls back to the deterministic template memo — the workflow
  never breaks because the LLM is unavailable.
"""
from __future__ import annotations

import json
import os


def _config() -> dict | None:
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        return None
    # Foundry project endpoints look like .../api/projects/<name>;
    # the chat-completions client wants the resource base URL.
    if "/api/projects/" in endpoint:
        endpoint = endpoint.split("/api/projects/")[0]
    return {
        "key": key,
        "endpoint": endpoint,
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
    }


def llm_available() -> bool:
    if _config() is None:
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


SYSTEM_PROMPT = """You are a credit operations writer at a commercial lender.
You will receive (1) a structured compliance evaluation produced by a
deterministic rules engine, and (2) a fact-sheet memo listing the findings.

Rewrite the fact sheet as a professional review memo in flowing prose.

Hard rules:
- Use ONLY facts present in the input. Do not add, infer, soften, or
  re-decide anything. The decision, severities, and routing are final.
- Preserve every exception code (e.g. E-141), policy section number,
  observed value, threshold, and waiver-authority statement exactly.
- If a finding is marked not waivable, state that plainly.
- 150 words maximum. No headers, no bullet points. Begin with the loan ID
  and decision. End with: "Facts per rules engine; prose drafted by LLM."
"""


def llm_memo_drafter(state: dict) -> str:
    """memo_drafter implementation: LLM prose with automatic fallback."""
    from agent.review_agent import template_memo  # local import: no cycle at module load

    facts = template_memo(state)
    cfg = _config()
    if cfg is None:
        return facts

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=cfg["key"],
            azure_endpoint=cfg["endpoint"],
            api_version="2024-10-21",
        )
        payload = {
            "evaluation": {
                k: v for k, v in state["explanation"].items() if k != "findings"
            },
            "decision": state.get("decision"),
            "queue": state.get("queue"),
            "minimum_authority": state.get("minimum_authority"),
            "blocking_codes": state.get("blocking_codes"),
            "findings": [
                {k: v for k, v in f.items() if k != "policy_text"}
                for f in state["explanation"]["findings"]
            ],
            "fact_sheet_memo": facts,
        }
        resp = client.chat.completions.create(
            model=cfg["deployment"],
            temperature=0.2,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else facts
    except Exception as e:  # any failure -> deterministic fallback
        return facts + f"\n\n(LLM drafter unavailable, template used: {type(e).__name__})"
