"""
GST Compliance Gym — Baseline Inference Script
===============================================
Hackathon evaluators run this script to benchmark an LLM agent against
the three task difficulties (easy, medium, hard).

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \
    HF_TOKEN=hf_xxx \
    MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
    python3 inference.py
"""

import os
import re
import json
import textwrap
from typing import List, Optional, Dict

from openai import OpenAI

try:
    from client import GSTComplianceEnv
    from models import CallToolAction
except ImportError:
    from gst_compliance_gym.client import GSTComplianceEnv
    from gst_compliance_gym.models import CallToolAction


# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS: int = 50
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 500
FALLBACK_ACTION: str = "get_invoices()"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are a GST (Goods and Services Tax) compliance auditor AI assistant.
    Your job is to audit a set of invoices for a business, identify compliance
    issues, compute taxes, and submit a GST return.

    You have access to the following 8 tools. Call exactly ONE tool per message,
    using the exact syntax shown below:

    1.  get_invoices()
        — List all invoices for the current episode.

    2.  get_invoice_details(invoice_id="INV-001")
        — Fetch full details for a specific invoice.

    3.  validate_gstin_tool(gstin="29AALCS0297D1ZE")
        — Check whether a GSTIN is structurally valid and active.

    4.  classify_hsn(product_description="wireless bluetooth earbuds")
        — Classify a product and return the correct HSN code.

    5.  compute_tax_tool(invoice_id="INV-001", hsn_code="85183000", place_of_supply="29")
        — Compute CGST/SGST/IGST for an invoice given its HSN and place of supply.

    6.  reconcile_invoice(invoice_id="INV-001")
        — Reconcile an invoice against GSTR-2B data (available in hard mode only).

    7.  flag_invoice(invoice_id="INV-001", reason="invalid_gstin")
        — Flag an invoice as non-compliant. Valid reasons:
          invalid_gstin, missing_fields, wrong_hsn, tax_mismatch,
          itc_ineligible, circular_trading, duplicate_invoice

    8.  submit_return(return_data='{"total_taxable_value":0,"total_cgst":0,"total_sgst":0,"total_igst":0}')
        — Submit the final GST return and end the episode.

    Strategy:
    - Start by calling get_invoices() to see all invoices.
    - Use get_invoice_details() to inspect each invoice in turn.
    - Validate the vendor GSTIN with validate_gstin_tool().
    - Use classify_hsn() when the HSN code looks wrong or is missing.
    - Use compute_tax_tool() to verify tax calculations.
    - In hard mode, use reconcile_invoice() to check GSTR-2B matching.
    - Flag every invoice that has a compliance issue using flag_invoice().
    - Once you have reviewed all invoices, call submit_return() with the
      aggregated totals.

    IMPORTANT: Respond with EXACTLY ONE tool call per message, nothing else.
    Format: tool_name(key="value", key2="value2")
""")


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

def parse_tool_call(response_text: str) -> Optional[Dict]:
    """
    Parse a tool call of the form:
        tool_name(key="value", key2="value2")

    Returns a dict like:
        {"tool_name": "get_invoice_details", "arguments": {"invoice_id": "INV-001"}}

    Returns None if no valid tool call is found.
    """
    # Strip whitespace and code-fence markers
    text = response_text.strip()
    text = re.sub(r"```[a-zA-Z]*", "", text).strip("`").strip()

    # Match: word_chars( ... )
    pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)"
    match = re.search(pattern, text)
    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    arguments: Dict[str, str] = {}
    if args_str:
        # Match key="value" or key='value' pairs
        kv_pattern = r"""([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')"""
        for kv_match in re.finditer(kv_pattern, args_str):
            key = kv_match.group(1)
            # group(2) for double-quoted, group(3) for single-quoted
            value = kv_match.group(2) if kv_match.group(2) is not None else kv_match.group(3)
            arguments[key] = value

    return {"tool_name": tool_name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(step: int, observation, history: List[str]) -> str:
    """
    Build the user-turn prompt from the current observation and recent history.

    observation — the result returned by env.reset() or env.step()
    history     — list of recent "Step N: tool(...) → result" strings
    """
    # Serialize observation metadata as JSON
    metadata: dict = {}
    if hasattr(observation, "metadata") and observation.metadata:
        metadata = observation.metadata
    if hasattr(observation, "result") and observation.result is not None:
        try:
            result_text = observation.result.content[0].text
            metadata["last_tool_result"] = json.loads(result_text)
        except (AttributeError, IndexError, json.JSONDecodeError, TypeError):
            try:
                metadata["last_tool_result"] = str(observation.result)
            except Exception:
                pass

    observation_json = json.dumps(metadata, indent=2, ensure_ascii=False)

    # Last 6 history entries
    recent_history = history[-6:] if len(history) > 6 else history
    history_text = "\n".join(recent_history) if recent_history else "(none yet)"

    prompt = textwrap.dedent(f"""\
        Step {step} of {MAX_STEPS}.

        Current observation:
        {observation_json}

        Recent actions and results:
        {history_text}

        What tool do you call next? Respond with exactly one tool call.
    """)
    return prompt


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the baseline agent on all three tasks and print scores."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = GSTComplianceEnv.from_docker_image(image="gst-compliance-gym:latest")

    scores: Dict[str, float] = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"Task: {task_id.upper()}")
        print(f"{'='*60}")

        result = env.reset(seed=42, task_id=task_id)
        observation = result
        history: List[str] = []
        last_reward: float = 0.0

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            user_prompt = build_user_prompt(step, observation, history)

            # Call the LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = response.choices[0].message.content or ""
            except Exception as exc:
                print(f"  Step {step}: LLM error — {exc}")
                response_text = FALLBACK_ACTION

            # Parse tool call
            parsed = parse_tool_call(response_text)
            if parsed is None:
                print(f"  Step {step}: could not parse tool call from: {response_text!r}")
                # Fall back to a safe no-op tool call
                parsed = parse_tool_call(FALLBACK_ACTION)

            tool_name: str = parsed["tool_name"]  # type: ignore[index]
            arguments: Dict[str, str] = parsed["arguments"]  # type: ignore[index]

            print(f"  Step {step}: {tool_name}({arguments})")

            # Step the environment
            try:
                observation = env.step(
                    CallToolAction(tool_name=tool_name, arguments=arguments)
                )
                last_reward = observation.reward
            except Exception as exc:
                print(f"  Step {step}: env.step() error — {exc}")
                break

            # Record history
            history_entry = f"Step {step}: {tool_name}({arguments}) → reward={last_reward:.4f}"
            history.append(history_entry)

            if observation.done:
                print(f"  Episode done at step {step}.")
                break

        scores[task_id] = last_reward
        print(f"Task {task_id}: score = {last_reward:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id:<10} score = {score:.4f}")
    total = sum(scores.values())
    print(f"  {'TOTAL':<10} score = {total:.4f}")


if __name__ == "__main__":
    main()
