# ========================
# prompts.py
# ========================
# Orchestrates communication with the LLM (via Groq).
# - Builds structured messages
# - Enforces English-only explanations
# - Adjusts tone depending on client persona (Conservative / Aggressive / Balanced)
# - Applies formatting rules (headings, bullet points, evidence lines)
# - NEW: Guarantees EVERY recommendation from the engine is rendered (no dropping)
# - NEW: Adds a deterministic fallback (no LLM) so nothing gets lost if the API fails
# ========================

import json
import os
from typing import Dict, Any, List

from groq import Groq  # pip install groq


# -----------------------
# Message builder (always English)
# -----------------------
def build_messages(portfolio: Dict, engine_output: Dict) -> List[Dict]:
    """
    Build structured chat messages for the LLM.
    - Always outputs in English
    - Includes persona-specific tone guidance
    - Provides formatting instructions
    - NEW: Explicitly require that ALL recommendations in the JSON are rendered (no omission)
    """
    persona = engine_output.get("risk_persona", "Balanced")

    # System role: high-level rules for the assistant
    system = {
        "role": "system",
        "content": (
            "You are an investment assistant specialized in the Egyptian Stock Exchange (EGX). "
            "Explain recommendations clearly and concisely in English. "
            "Keep tone professional, factual, and educational. "
            "Do not guarantee returns or provide financial advice. "
            "Flag stale data if indicated. "
            "Output structure:\n"
            "1) One-line summary of the client's risk persona.\n"
            "2) Structural recommendations first (sector concentration, diversification).\n"
            "3) Tactical recommendations next (top movers, watchlist drops).\n"
            "4) Data quality notes last.\n"
            "Each recommendation: short title + one-sentence explanation. "
            "Then provide an evidence line. "
            "If stale==true, append '(data may be outdated)'.\n\n"
            # Persona-aware tone guidance
            f"For persona adaptation: If persona is Conservative, phrase cautiously "
            f"(e.g., 'consider gradually reducing exposure'). "
            f"If Aggressive, highlight opportunities. "
            f"Balanced should mix both tones.\n\n"
            # NEW: No omissions
            "IMPORTANT COMPLETENESS RULES:\n"
            "- You MUST include EVERY recommendation item from the JSON payload.\n"
            "- Do NOT drop, merge, or summarize away any item.\n"
            "- Preserve the order of items as provided when possible.\n"
            "- If there are many items, keep bullets concise but include them all."
        )
    }

    # Developer role: formatting & output rules
    developer = {
        "role": "system",
        "content": (
            "Formatting rules:\n"
            "- Use headings for sections (Summary, Recommendations, Data Notes).\n"
            "- Use bullet points for recommendations; one bullet per recommendation.\n"
            "- Include a short evidence line per recommendation (stock, sector, % change, etc.).\n"
            "- Keep language concise. Prioritize completeness over brevity.\n"
            "- NEVER omit or collapse items from the provided list."
        )
    }

    # User role: actual payload (portfolio + recs JSON)
    user_payload = {
        "client": {
            "id": engine_output.get("client_id") or portfolio.get("clientid"),
            "name": portfolio.get("clientnamee"),
            "risk_persona": persona,
            "persona_confidence": engine_output.get("persona_confidence"),
        },
        "recommendations": engine_output.get("recommendations", []),
        "meta": engine_output.get("meta", {})
    }

    user = {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}

    return [system, developer, user]


# -----------------------
# Groq chat call (free tier available)
# -----------------------
def render_with_llm(
    messages: List[Dict],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 1200,  # ↑ bump to reduce truncation when many items exist
) -> str:
    """
    Calls Groq to turn the engine JSON into clean English advice.

    Setup:
      - pip install groq
      - set GROQ_API_KEY env var
      - default model: llama-3.1-8b-instant
    """
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a free key at console.groq.com, then set the "
            "environment variable, e.g. PowerShell:\n\n"
            '  setx GROQ_API_KEY "your_key_here"\n'
            "Restart your terminal afterwards."
        )

    client = Groq(api_key=api_key)
    chosen_model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# -----------------------
# Deterministic fallback (no LLM) to guarantee completeness
# -----------------------
def _fallback_render_all(engine_output: Dict) -> str:
    """
    If the LLM call fails or times out, render EVERY recommendation deterministically.
    This guarantees the UI still shows a complete list.
    """
    client = engine_output.get("client_id", "Unknown")
    persona = engine_output.get("risk_persona", "Balanced")
    conf = engine_output.get("persona_confidence", "N/A")
    recs = engine_output.get("recommendations", [])
    meta = engine_output.get("meta", {})

    lines = []
    lines.append(f"**Summary**")
    lines.append(f"Client's risk persona: {persona} (confidence level: {conf})")
    ts = meta.get("data_timestamp")
    if ts:
        lines.append(f"Data as of: {ts}")
    lines.append("")  # spacer

    lines.append("**Recommendations**")
    if not recs:
        lines.append("- No recommendations available.")
    else:
        for r in recs:
            title = r.get("type", "item").replace("_", " ").title()
            msg = r.get("message", "").strip()
            stale = r.get("stale", False)
            suffix = " (data may be outdated)" if stale else ""
            lines.append(f"- **{title}**: {msg}{suffix}")
            ev = r.get("evidence", {})
            if ev:
                # compact evidence line
                ev_str = ", ".join(f"{k}: {v}" for k, v in ev.items())
                lines.append(f"  - Evidence: {ev_str}")
    lines.append("")  # spacer

    # Data notes (if any data_quality recs exist they are already included above)
    dq = [r for r in recs if r.get("type") == "data_quality"]
    if dq:
        lines.append("**Data Notes**")
        for d in dq:
            issues = d.get("evidence", {}).get("issues", [])
            if issues:
                lines.append(f"- {', '.join(issues)}")

    return "\n".join(lines)


# -----------------------
# Main orchestrator
# -----------------------
def generate_advice(portfolio: Dict, engine_output: Dict) -> Dict[str, Any]:
    """
    Orchestrates prompts and LLM to generate the chatbot's final output.
    Returns dict with:
      - message_text (clean prose that includes EVERY recommendation)
      - engine_output (raw JSON, useful for debugging/UI)
    """
    messages = build_messages(portfolio, engine_output)

    # Try LLM; if it fails, fallback to deterministic renderer that shows ALL items.
    try:
        prose = render_with_llm(messages)
    except Exception:
        prose = _fallback_render_all(engine_output)

    return {
        "message_text": prose,
        "engine_output": engine_output
    }


# -----------------------
# Standalone test
# -----------------------
if __name__ == "__main__":
    import random
    import pandas as pd
    from recommendation_engine import generate_recommendations

    # Load any real client (random to vary tests)
    with open("Clients_Portfolios.json", "r", encoding="utf-8") as f:
        portfolios = json.load(f)
    portfolio = random.choice(portfolios)
    print(f"🔎 Testing with client: {portfolio.get('clientid')}")

    # Load market data
    market_df = pd.read_csv("Egypt_Equities.csv")

    # Get structured recs JSON from the engine
    engine_output = generate_recommendations(
        portfolio=portfolio,
        market=market_df,
        max_items=None,              # show all
        freshness_policy="degrade",  # label/penalize stale
        stale_after_minutes=120,
        market_asof=None
    )

    # Build messages (debug print)
    msgs = build_messages(portfolio, engine_output)
    print("\n🔎 Messages passed to LLM:")
    for m in msgs:
        preview = m["content"]
        if isinstance(preview, str) and len(preview) > 250:
            preview = preview[:250] + " ..."
        print(f"{m['role']}: {preview}\n")

    # Render with Groq (with guaranteed fallback)
    try:
        advice_text = render_with_llm(
            msgs,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.3,
            max_tokens=1200
        )
    except Exception as e:
        print("\n⚠️ LLM call failed, using fallback renderer:", e)
        advice_text = _fallback_render_all(engine_output)

    print("\n💬 Final advice:\n")
    print(advice_text)
