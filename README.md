---
title: GST Compliance Gym
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# GST Compliance Gym

An [OpenEnv](https://github.com/openenv/openenv) environment for Indian GST compliance auditing.
A hackathon-ready RL benchmark where an LLM agent audits synthetic invoices,
validates GSTINs, classifies HSN codes, computes taxes, reconciles GSTR-2B
records, and flags fraud — all through 8 deterministic MCP tools.

---

## Motivation

India's GST system is one of the largest tax administration networks in the
world, covering **14 million+ registered businesses** and generating hundreds
of millions of invoices every month.  Compliance auditing is an ideal
reinforcement-learning environment because:

- **Deterministic rewards** — every action has a ground-truth answer derived
  from the GST Act, HSN schedule, and state-code rules.
- **Multi-step reasoning** — a correct audit requires 8–20 tool calls in a
  specific logical order (inspect → validate → classify → compute → reconcile →
  flag → submit).
- **LLM-hard** — the agent must parse structured documents, perform arithmetic,
  understand legal concepts (ITC, GSTR-2B, place of supply), and detect
  adversarial fraud patterns such as circular trading.
- **Partial credit** — scoring rewards incremental progress so gradient signal
  exists even for imperfect agents.
- **Scalable difficulty** — three task tiers from basic GSTIN validation to
  full fraud detection, making it suitable for weak and strong model evaluation.

---

## Environment Description

Each episode represents one **tax-period audit** for a synthetic Indian company.
The environment generates a fresh set of invoices, vendors, GSTINs, HSN codes,
and (in hard mode) GSTR-2B records and fraud invoices — all seeded for
reproducibility.

### Episode flow

1. `reset(task_id=...)` — loads a new episode; returns the business profile and
   task instructions in the observation metadata.
2. The agent calls MCP tools (one per step) to inspect and audit the invoices.
3. The agent calls `submit_return(...)` to finalise the audit and end the
   episode.
4. A deterministic grader computes a normalised score in `[0.0, 1.0]`.

Max steps per episode: **50**. Exceeding the limit terminates the episode with
a `−0.1` penalty.

---

## Action Space — MCP Tools

All actions are MCP tool calls.  The agent calls exactly one tool per step.

| # | Tool name | Description | Parameters |
|---|-----------|-------------|------------|
| 1 | `get_invoices` | List all invoice summaries for the episode | _(none)_ |
| 2 | `get_invoice_details` | Fetch full details for a specific invoice | `invoice_id: str` |
| 3 | `validate_gstin_tool` | Validate a GSTIN string (structure + Luhn mod-36 checksum) | `gstin: str` |
| 4 | `classify_hsn` | Keyword-match a product description to an HSN code and tax rate | `product_description: str` |
| 5 | `compute_tax_tool` | Compute CGST/SGST/IGST for an invoice given its HSN and place of supply | `invoice_id: str`, `hsn_code: str`, `place_of_supply: str` |
| 6 | `reconcile_invoice` | Match an invoice against GSTR-2B records (hard mode only) | `invoice_id: str` |
| 7 | `flag_invoice` | Flag an invoice as non-compliant or fraudulent | `invoice_id: str`, `reason: str` |
| 8 | `submit_return` | Submit the final GST return; ends the episode and triggers grading | `return_data: str` (JSON) |

Valid `reason` values for `flag_invoice`:
`invalid_gstin`, `missing_fields`, `wrong_hsn`, `tax_mismatch`,
`itc_ineligible`, `circular_trading`, `duplicate_invoice`.

---

## Observation Space

Every `step()` call returns an `Observation` object with:

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Per-step reward for the last action |
| `metadata.step_count` | `int` | Current step number |
| `metadata.task_id` | `str` | Active task (`easy`, `medium`, `hard`) |
| `metadata.last_tool_result` | `dict` | JSON result of the last MCP tool call |

The initial observation from `reset()` also includes:

| Field | Description |
|-------|-------------|
| `metadata.business` | Company name, GSTIN, state, state code, tax period |
| `metadata.invoice_count` | Total number of invoices to audit |
| `metadata.instructions` | Natural-language task instructions for the agent |

---

## Tasks

### Easy — Invoice Validation

| Parameter | Value |
|-----------|-------|
| Invoices | 5 |
| Invalid invoices | 2 (invalid GSTIN or missing fields) |
| Expected score range | 0.6 – 0.8 |
| Key tools | `get_invoices`, `get_invoice_details`, `validate_gstin_tool`, `flag_invoice`, `submit_return` |

The agent must inspect all invoices, validate vendor GSTINs, and flag every
invalid invoice with the correct error reason.  Partial credit is awarded for
correct flags with wrong reasons (+0.10 vs +0.25 per flag).  False positives
deduct −0.15 per incorrect flag.

### Medium — Tax Computation & HSN Classification

| Parameter | Value |
|-----------|-------|
| Invoices | 10 |
| Wrong-HSN invoices | 3 (incorrect HSN codes causing tax rate errors) |
| Expected score range | 0.3 – 0.5 |
| Key tools | `classify_hsn`, `compute_tax_tool`, `flag_invoice`, `submit_return` |

The agent must verify that each invoice's HSN code is correct, reclassify
products where necessary, recompute CGST/SGST/IGST, and flag discrepancies.
Scoring is per-invoice: +0.04 for exact 8-digit HSN match, +0.025 for 6-digit
prefix match, +0.02 for correct interstate determination, +0.04 for total tax
within ₹1 of correct.

### Hard — ITC Reconciliation & Fraud Detection

| Parameter | Value |
|-----------|-------|
| Invoices | 15 (plus hidden fraud invoices) |
| ITC mismatches | 3–4 (GSTR-2B discrepancies) |
| Fraud invoices | 2–3 (circular trading pattern) |
| Expected score range | 0.1 – 0.3 |
| Key tools | `reconcile_invoice`, `flag_invoice`, `submit_return` |

The agent must reconcile every invoice against GSTR-2B data, identify ITC
mismatches, detect fraud invoices (circular trading network), and flag them
with the correct pattern.  Score is split equally between ITC reconciliation
accuracy (capped at 0.5) and fraud detection accuracy (capped at 0.5).

---

## Reward Design

Rewards are **dense** (awarded on every step) and **deterministic**.

| Event | Reward |
|-------|--------|
| `get_invoices()` called | +0.01 |
| `get_invoice_details()` for a valid invoice | +0.01 |
| `validate_gstin_tool()` — valid GSTIN found | +0.02 |
| `validate_gstin_tool()` — invalid GSTIN found | +0.03 |
| `classify_hsn()` — match found | +0.05 |
| `classify_hsn()` — no match | −0.02 |
| `compute_tax_tool()` — successful computation | +0.05 |
| `compute_tax_tool()` — invoice or HSN not found | −0.03 |
| `reconcile_invoice()` — mismatch detected | +0.05 |
| `reconcile_invoice()` — clean match confirmed | +0.02 |
| `reconcile_invoice()` — invoice not found | −0.03 |
| `flag_invoice()` — correct flag (valid issue) | +0.08 |
| `flag_invoice()` — correct fraud flag | +0.10 |
| `flag_invoice()` — false positive | −0.05 |
| `submit_return()` | Final grader score (0.0 – 1.0) |
| Repeated identical tool call | −0.01 |
| Steps exceeded (> 50) | −0.1, episode terminates |

Anti-gaming properties:
- Repeated identical calls are penalised.
- False-positive flags deduct score.
- HSN misclassification with wrong-chain codes receives partial (not full) credit.
- Circular trading detection requires naming the pattern explicitly in the
  `flag_invoice` reason.

---

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment or `from_docker_image` client)

### Install

```bash
git clone https://github.com/your-org/gst-compliance-gym
cd gst-compliance-gym

# Install with uv (recommended)
uv pip install -e ".[dev,inference]"

# Or with pip
pip install -e ".[dev,inference]"
```

### Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker

```bash
cd server
docker build -t gst-compliance-gym:latest .
docker run -p 8000:8000 gst-compliance-gym:latest
```

### Run the baseline inference script

Set the following environment variables, then run `inference.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | OpenAI-compatible API base URL | `https://router.huggingface.co/v1` |
| `HF_TOKEN` | Hugging Face API token (used as API key) | _(required)_ |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
HF_TOKEN=hf_xxx \
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
python3 inference.py
```

The script runs the agent on all three tasks (`easy`, `medium`, `hard`) and
prints per-task and total scores.

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
gst-compliance-gym/
├── inference.py            # Baseline LLM agent + evaluation loop
├── client.py               # GSTComplianceEnv client wrapper
├── models.py               # Pydantic types (CallToolAction, etc.)
├── openenv.yaml            # OpenEnv spec (name, runtime, port)
├── pyproject.toml          # Package metadata and dependencies
├── server/
│   ├── app.py              # FastAPI application entry point
│   ├── gst_environment.py  # Core MCPEnvironment with 8 tools (MAX_STEPS=50)
│   ├── gst_rules.py        # GSTIN validator, HSN database (69 entries), tax engine
│   ├── data_generator.py   # Seeded synthetic episode generator
│   └── graders.py          # Deterministic graders for easy/medium/hard
└── tests/                  # pytest test suite
```

---

## Baseline Scores

Scores to be filled after running `inference.py` with each model.

| Model | Easy | Medium | Hard | Total |
|-------|------|--------|------|-------|
| `meta-llama/Llama-3.3-70B-Instruct` | TBD | TBD | TBD | TBD |
| `mistralai/Mistral-7B-Instruct-v0.3` | TBD | TBD | TBD | TBD |
| Random baseline | TBD | TBD | TBD | TBD |

---

## HSN Database

The environment includes **69 HSN/SAC codes** across the following categories:

| Category | Tax rates covered |
|----------|-------------------|
| Food & Beverages | 0%, 5%, 12%, 18%, 28% |
| Textiles | 5%, 12% |
| Chemicals & Pharma | 12%, 18% |
| Plastics & Rubber | 18%, 28% |
| Iron & Steel | 18% |
| Machinery & Electronics | 12%, 18%, 28% |
| Vehicles | 28% |
| Furniture | 18%, 28% |
| Services (SAC codes) | 5%, 18% |

All 37 Indian state and union territory codes are supported for place-of-supply
and GSTIN validation (codes `01`–`37`).

---

## License

Apache 2.0
