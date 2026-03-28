# GST Compliance Gym — Design Spec

## Overview

An OpenEnv environment that simulates Indian GST (Goods & Services Tax) compliance auditing. An AI agent acts as a compliance auditor for an Indian business, validating invoices, classifying products, computing taxes, reconciling ITC claims, detecting fraud, and preparing GSTR returns.

**Target hackathon:** Meta x OpenEnv Hackathon by Scaler (Round 1, deadline April 8, 2026)
**HF Space:** `lilbabycrypto/gst-compliance-gym`
**Architecture:** MCP Tool-based (`MCPEnvironment` + `FastMCP`)

## Why This Domain

- 14M GST-registered businesses in India
- Rs 1.79 lakh crore in fake ITC fraud detected (FY21-25)
- Tax math is objectively verifiable — deterministic, reproducible rewards
- Multi-step reasoning required (validate → classify → compute → reconcile → file)
- LLMs struggle with precise arithmetic, structured compliance, and graph reasoning
- Zero existing OpenEnv implementations in tax/compliance domain

## Architecture

### Approach: MCP Tool-based

Uses `MCPEnvironment` with `FastMCP` tools. The LLM agent discovers tools via `list_tools()` and calls them through function calling. This is the architecture the OpenEnv reference environment (echo) uses and aligns with how LLM agents naturally interact.

### Project Structure

```
gst-compliance-gym/
  __init__.py
  README.md
  client.py
  models.py
  openenv.yaml
  pyproject.toml
  uv.lock
  inference.py              # Root level, hackathon requirement
  server/
    __init__.py
    app.py
    gst_environment.py      # Core environment logic (MCPEnvironment subclass)
    data_generator.py       # Synthetic invoice/business generator
    gst_rules.py            # GSTIN validation, HSN lookup, tax computation
    graders.py              # Task-specific grading logic
    Dockerfile
    .dockerignore
```

## MCP Tools (Agent Actions)

The agent discovers and calls these 8 tools:

### 1. `get_invoices()`
- **Purpose:** List all invoices for the tax period
- **Returns:** List of invoice summaries (id, vendor_name, amount, invoice_type B2B/B2C, date)
- **Reward:** +0.01

### 2. `get_invoice_details(invoice_id: str)`
- **Purpose:** Get full details of a specific invoice
- **Returns:** All 15+ mandatory fields — GSTIN, HSN code, product description, taxable value, tax amounts, place of supply, date, invoice number, etc.
- **Reward:** +0.01

### 3. `validate_gstin(gstin: str)`
- **Purpose:** Validate GSTIN format and Luhn mod-36 checksum
- **Returns:** `{valid: bool, reason: str}` — reason includes specific failure (bad checksum, wrong state code, invalid PAN format)
- **Reward:** +0.02 (valid GSTIN correctly confirmed), +0.03 (invalid GSTIN correctly caught)

### 4. `classify_hsn(product_description: str)`
- **Purpose:** Look up the correct HSN/SAC code for a product/service description
- **Returns:** `{hsn_code: str, description: str, tax_rate: float}` from the internal HSN database
- **Reward:** +0.05 (exact 8-digit match), +0.02 (correct 6-digit prefix), -0.02 (wrong chapter)

### 5. `compute_tax(invoice_id: str, hsn_code: str, place_of_supply: str)`
- **Purpose:** Calculate CGST/SGST or IGST based on HSN rate and place of supply vs business state
- **Returns:** `{cgst: float, sgst: float, igst: float, cess: float, total_tax: float, is_interstate: bool}`
- **Reward:** +0.05 (correct amount within Rs 1), -0.03 (wrong amount)

### 6. `reconcile_invoice(invoice_id: str)`
- **Purpose:** Match invoice against supplier's GSTR-2B data for ITC eligibility
- **Returns:** `{match_status: "matched"|"mismatched"|"missing", discrepancies: [{field, expected, actual}], itc_eligible: bool}`
- **Reward:** +0.05 (correct match/mismatch decision), -0.03 (wrong decision)

### 7. `flag_invoice(invoice_id: str, reason: str)`
- **Purpose:** Flag an invoice as erroneous or fraudulent
- **Reason values:** `"invalid_gstin"`, `"missing_fields"`, `"wrong_hsn"`, `"tax_mismatch"`, `"itc_ineligible"`, `"circular_trading"`, `"duplicate_invoice"`
- **Returns:** `{accepted: bool, correct: bool, feedback: str}`
- **Reward:** +0.08 (true positive), +0.10 (correct fraud detection), -0.05 (false positive)

### 8. `submit_return(return_data: dict)`
- **Purpose:** Submit the final GSTR summary, ending the episode
- **Input:** `{total_taxable_value, total_cgst, total_sgst, total_igst, total_cess, flagged_invoices: [ids], itc_claimed: float}`
- **Returns:** `{score: float, breakdown: dict, ground_truth: dict}`
- **Reward:** Task grader score (0.0-1.0)
- **Side effect:** Sets `done=True`, episode ends

### Penalty Signals
- Repeated tool call on same invoice with same params: -0.01
- Calling `submit_return` without inspecting any invoices: -0.2

## Tasks & Graders

### Task 1: Invoice Validation (Easy)

- **Scenario:** 5 invoices, 2 have field-level errors (missing place of supply, invalid GSTIN checksum, wrong date format)
- **Objective:** Identify and flag the invalid invoices with correct reasons
- **Grader:**
  - Each correct flag with correct reason: +0.25
  - Correct flag, wrong reason: +0.10
  - Missed invalid invoice: +0.0
  - Flagging valid invoice (false positive): -0.15
  - Score = max(0.0, sum / max_possible), capped at 1.0
- **Expected LLM score:** 0.6-0.8

### Task 2: Tax Computation & HSN Classification (Medium)

- **Scenario:** 10 invoices with product descriptions. Agent must classify HSN, determine inter/intra-state, compute correct tax amounts
- **Objective:** Compute correct tax for each invoice and submit accurate GSTR-1 summary
- **Grader (per invoice, 0.1 each):**
  - Correct HSN 8-digit match: 0.04
  - Correct HSN 6-digit prefix only: 0.025
  - Correct inter/intra-state determination: 0.02
  - Correct tax amount (within Rs 1 tolerance): 0.04
  - Score = sum of per-invoice scores, capped at 1.0
- **Expected LLM score:** 0.3-0.5

### Task 3: ITC Reconciliation & Fraud Detection (Hard)

- **Scenario:** 15 invoices. Supplier GSTR-2B data available. 3 invoices have ITC mismatches (amount differs, invoice missing from supplier side, duplicate claim). 2 invoices are part of a circular trading pattern across 3 shell entities.
- **Objective:** Reconcile all invoices, identify mismatches with reasons, detect the fraud ring
- **Grader:**
  - ITC reconciliation (0.5 total):
    - Each correct match/mismatch decision: +0.08 (6 reconciliation checks)
    - Each correct mismatch reason: +0.04 (3 mismatches with reasons)
  - Fraud detection (0.5 total):
    - Identified both fraudulent invoices: +0.25
    - Identified the circular trading pattern: +0.20
    - Partial (found 1 of 2 fraudulent invoices): +0.10
    - False fraud flag on legitimate invoice: -0.10
  - Score = max(0.0, sum), capped at 1.0
- **Expected LLM score:** 0.1-0.3

### Difficulty Progression

| Task | Invoices | Error Types | Key Challenge | Expected Score |
|------|----------|-------------|---------------|---------------|
| Easy | 5 | Field errors | Pattern matching + checksum | 0.6-0.8 |
| Medium | 10 | HSN + tax math | Classification + arithmetic | 0.3-0.5 |
| Hard | 15 | ITC mismatch + fraud ring | Reconciliation + graph reasoning | 0.1-0.3 |

## Reward System

### Dense Per-Step Rewards

Every tool call produces immediate reward:

| Tool Call Result | Reward |
|-----------------|--------|
| `get_invoices` | +0.01 |
| `get_invoice_details(id)` | +0.01 |
| `validate_gstin` — valid confirmed | +0.02 |
| `validate_gstin` — invalid caught | +0.03 |
| `classify_hsn` — exact 8-digit match | +0.05 |
| `classify_hsn` — correct chapter, wrong sub | +0.02 |
| `classify_hsn` — wrong chapter | -0.02 |
| `compute_tax` — correct amount | +0.05 |
| `compute_tax` — wrong amount | -0.03 |
| `reconcile_invoice` — correct decision | +0.05 |
| `reconcile_invoice` — wrong decision | -0.03 |
| `flag_invoice` — true positive | +0.08 |
| `flag_invoice` — false positive | -0.05 |
| `flag_invoice` — correct fraud detection | +0.10 |
| Repeated call, same invoice + params | -0.01 |

### Episode-End Reward

`submit_return()` triggers the task grader which computes a final score (0.0-1.0). This is the **reported score** used by the hackathon evaluation. Step rewards provide training signal but the grader score is what matters for judging.

### Properties

- **Deterministic:** Same seed + same actions = same rewards, always
- **Dense:** Signal at every step, not just episode end
- **Hierarchical:** Partial credit at field/digit/invoice levels
- **Balanced:** Positive for correct, negative for harmful actions
- **Anti-gaming:** Repeated calls penalized, false positives penalized

## State Model

```python
State:
  episode_id: str           # Unique episode identifier
  step_count: int           # Current step number
  task_id: str              # "easy", "medium", "hard"
  invoices_viewed: int      # Number of invoices inspected
  validations_done: int     # Number of GSTINs validated
  flags_raised: int         # Number of invoices flagged
  reconciliations_done: int # Number of ITC reconciliations
  return_submitted: bool    # Whether episode has ended
```

## Episode Boundaries

- **Start:** `reset(seed=N, task_id="easy"|"medium"|"hard")` generates fresh business data
- **End:** Agent calls `submit_return()` OR `step_count` reaches 50
- **Timeout:** At 50 steps without submission, score = partial work done - 0.1 penalty
- **Seeded reproducibility:** Same seed produces identical invoice sets and errors

## Synthetic Data Generation

Each `reset()` procedurally generates:

1. **Business profile:** Company name, GSTIN (valid, Luhn mod-36), state, industry sector, turnover bracket
2. **Invoices:** N invoices with realistic fields:
   - Vendor names (from pool of Indian business names)
   - Product descriptions (from pool mapped to HSN codes)
   - Amounts (realistic ranges for the product category)
   - Dates within the tax period
   - Invoice numbers (sequential with occasional gaps)
3. **Embedded errors** (based on task):
   - Easy: Invalid GSTIN checksums, missing mandatory fields
   - Medium: Ambiguous product descriptions, inter/intra-state misclassification
   - Hard: ITC amount mismatches, missing supplier records, circular trading graph
4. **Supplier GSTR-2B data** (for hard task): Matching records with intentional discrepancies
5. **Ground truth:** Stored internally for deterministic grading

### HSN Database

Internal lookup table with ~200 common Indian product/service categories mapped to 8-digit HSN/SAC codes and tax rates (5%, 12%, 18%, 28%). Covers electronics, textiles, food, machinery, services.

### GSTIN Validation

Implements full Luhn mod-36 checksum:
- 2-digit state code (01-37)
- 10-digit PAN
- Entity code (1-9, A-Z)
- 'Z' fixed character
- Checksum digit

## Inference Script

`inference.py` in project root. Follows the hackathon sample pattern (text-completion style with action parsing):

### Environment Variables (mandatory)
- `API_BASE_URL` — LLM endpoint (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` — model identifier for inference
- `HF_TOKEN` — HuggingFace API key

### Flow

```python
# 1. Setup
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
env = GSTComplianceEnv.from_docker_image(image="gst-compliance-gym:latest")

# 2. For each task
for task_id in ["easy", "medium", "hard"]:
    result = env.reset(seed=42, task_id=task_id)
    observation = result.observation
    history = []

    # 3. Agent loop
    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        # Build prompt with observation + available tools
        user_prompt = build_user_prompt(step, observation, history)

        # LLM decides which tool to call
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[system_msg, {"role": "user", "content": user_prompt}],
            temperature=0.2, max_tokens=500
        )

        # Parse LLM response into CallToolAction
        action = parse_tool_call(completion.choices[0].message.content)

        # Step environment
        result = env.step(action)
        observation = result.observation

        history.append(f"Step {step}: {action} -> reward {result.reward}")

    print(f"Task {task_id}: score = {result.reward}")
```

### System Prompt
Describes the agent's role as a GST compliance auditor, lists available tools with descriptions, and instructs the LLM to respond with exactly one tool call per step in the format: `tool_name(arg1="value1", arg2="value2")`

### Action Parsing
Regex-based extraction of tool name and arguments from LLM text response, with fallback to `get_invoices()` if parsing fails.

### Constraints
- MAX_STEPS = 50 per task
- Total runtime < 20 minutes for all 3 tasks
- Runs on 2 vCPU, 8GB RAM

## Deployment

- **Dockerfile:** Based on `ghcr.io/meta-pytorch/openenv-base:latest`, deps via `uv sync`, runs uvicorn
- **HF Space:** `lilbabycrypto/gst-compliance-gym`, tagged `openenv`
- **Resources:** Runs on 2 vCPU, 8GB RAM (no external API calls, all synthetic data)
- **Deploy command:** `openenv push --repo-id lilbabycrypto/gst-compliance-gym`

## Validation Checklist

- [ ] `openenv.yaml` with `spec_version: 1`
- [ ] `pyproject.toml` with `openenv-core[core]>=0.2.2` dependency
- [ ] `uv.lock` generated
- [ ] `server/app.py` with `create_app()` and `main()`
- [ ] `server/Dockerfile` builds and runs
- [ ] `models.py` with proper imports
- [ ] `client.py` with MCP client
- [ ] `reset()` returns valid observation with HTTP 200
- [ ] `step()` processes tool calls correctly
- [ ] `state` property returns current state
- [ ] 3 tasks with graders producing 0.0-1.0 scores
- [ ] `inference.py` runs end-to-end and prints scores
- [ ] HF Space deploys and responds
- [ ] README with full documentation
