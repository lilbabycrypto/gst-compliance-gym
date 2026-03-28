# GST Compliance Gym — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete OpenEnv environment simulating Indian GST compliance auditing, deployable to HuggingFace Spaces, with 3 graded tasks and a baseline inference script.

**Architecture:** MCP Tool-based environment using `MCPEnvironment` + `FastMCP`. 8 MCP tools for GST operations. Synthetic data generation for reproducible episodes. 3 task graders (easy/medium/hard) with deterministic 0.0-1.0 scoring.

**Tech Stack:** Python 3.12, openenv-core, FastMCP, FastAPI, Pydantic, uvicorn, Docker, uv

---

## File Structure

```
gst-compliance-gym/
  __init__.py                    # Package exports: GSTComplianceEnv client, action types
  client.py                      # MCPToolClient subclass for external consumers
  models.py                      # Re-exports MCP action/observation types
  openenv.yaml                   # OpenEnv manifest (spec_version 1)
  pyproject.toml                 # Package config with deps and server entrypoint
  inference.py                   # Baseline agent script (hackathon requirement)
  README.md                      # Full documentation
  server/
    __init__.py                  # Exports GSTComplianceEnvironment
    app.py                       # FastAPI app via create_app()
    gst_environment.py           # MCPEnvironment subclass with 8 MCP tools
    data_generator.py            # Seeded synthetic invoice/business generator
    gst_rules.py                 # GSTIN validation, HSN database, tax computation
    graders.py                   # 3 task-specific deterministic graders
    Dockerfile                   # Container for HF Spaces deployment
    .dockerignore                # Exclude dev files from Docker build
  tests/
    __init__.py
    test_gst_rules.py            # Tests for GSTIN validation, HSN lookup, tax math
    test_data_generator.py       # Tests for invoice generation, error injection
    test_graders.py              # Tests for all 3 grading functions
    test_environment.py          # Integration tests for the full environment
```

---

### Task 1: Project Scaffold & Config

**Files:**
- Create: `gst-compliance-gym/openenv.yaml`
- Create: `gst-compliance-gym/pyproject.toml`
- Create: `gst-compliance-gym/__init__.py`
- Create: `gst-compliance-gym/models.py`
- Create: `gst-compliance-gym/client.py`
- Create: `gst-compliance-gym/server/__init__.py`
- Create: `gst-compliance-gym/server/app.py`
- Create: `gst-compliance-gym/server/gst_environment.py` (skeleton)
- Create: `gst-compliance-gym/server/Dockerfile`
- Create: `gst-compliance-gym/server/.dockerignore`
- Create: `gst-compliance-gym/tests/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git init
```

- [ ] **Step 2: Create openenv.yaml**

```yaml
spec_version: 1
name: gst_compliance_gym
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-gst-compliance-gym"
version = "0.1.0"
description = "OpenEnv environment for Indian GST compliance auditing"
requires-python = ">=3.10"
dependencies = [
    "openenv-core[core]>=0.2.2",
    "fastapi>=0.115.0",
    "pydantic>=2.0.0",
    "uvicorn>=0.24.0",
    "requests>=2.31.0",
    "fastmcp>=0.1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=4.0.0"]
inference = ["openai>=1.0.0"]

[project.scripts]
server = "gst_compliance_gym.server.app:main"

[tool.setuptools]
include-package-data = true
packages = ["gst_compliance_gym", "gst_compliance_gym.server"]
package-dir = {"gst_compliance_gym" = ".", "gst_compliance_gym.server" = "server"}
```

- [ ] **Step 4: Create models.py**

```python
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)

__all__ = [
    "CallToolAction",
    "CallToolObservation",
    "ListToolsAction",
    "ListToolsObservation",
]
```

- [ ] **Step 5: Create client.py**

```python
from openenv.core.mcp_client import MCPToolClient


class GSTComplianceEnv(MCPToolClient):
    pass
```

- [ ] **Step 6: Create __init__.py (root)**

```python
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from .client import GSTComplianceEnv

__all__ = ["GSTComplianceEnv", "CallToolAction", "ListToolsAction"]
```

- [ ] **Step 7: Create server/__init__.py**

```python
from .gst_environment import GSTComplianceEnvironment

__all__ = ["GSTComplianceEnvironment"]
```

- [ ] **Step 8: Create server/app.py (skeleton)**

```python
try:
    from ..models import CallToolAction, CallToolObservation
    from .gst_environment import GSTComplianceEnvironment
except (ImportError, ModuleNotFoundError):
    from models import CallToolAction, CallToolObservation
    from server.gst_environment import GSTComplianceEnvironment

from openenv.core.env_server.http_server import create_app

app = create_app(
    GSTComplianceEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="gst_compliance_gym",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Create server/gst_environment.py (skeleton)**

```python
from typing import Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, State


class GSTComplianceEnvironment(MCPEnvironment):
    """Indian GST compliance auditing environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        mcp = FastMCP("gst_compliance_gym")
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        return Observation(done=False, reward=0.0, metadata={"status": "ready"})

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False, reward=0.0, metadata={"error": "Unknown action"}
        )

    def step(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
```

- [ ] **Step 10: Create server/Dockerfile**

```dockerfile
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-install-project --no-editable; \
    else uv sync --no-install-project --no-editable; fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-editable; \
    else uv sync --no-editable; fi

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
```

- [ ] **Step 11: Create server/.dockerignore**

```
__pycache__
*.pyc
.git
.venv
tests/
docs/
*.md
```

- [ ] **Step 12: Create tests/__init__.py**

Empty file.

- [ ] **Step 13: Generate uv.lock and verify scaffold**

```bash
cd /Users/laptopfirst/gst-compliance-gym
uv lock
uv sync
```

- [ ] **Step 14: Commit scaffold**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add -A
git commit -m "feat: project scaffold with OpenEnv config, models, client, server skeleton"
```

---

### Task 2: GST Rules Engine

**Files:**
- Create: `gst-compliance-gym/server/gst_rules.py`
- Create: `gst-compliance-gym/tests/test_gst_rules.py`

- [ ] **Step 1: Write failing tests for GSTIN validation**

```python
# tests/test_gst_rules.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.gst_rules import validate_gstin, generate_valid_gstin, HSN_DATABASE, compute_tax


class TestGSTINValidation:
    def test_valid_gstin_returns_valid(self):
        # 29AALCS0297D1ZE is a known valid format GSTIN
        gstin = generate_valid_gstin(state_code="29", pan="AALCS0297D", entity="1")
        result = validate_gstin(gstin)
        assert result["valid"] is True
        assert result["reason"] == "Valid GSTIN"

    def test_wrong_length_returns_invalid(self):
        result = validate_gstin("29AALCS0297D1Z")
        assert result["valid"] is False
        assert "15 characters" in result["reason"]

    def test_invalid_state_code_returns_invalid(self):
        result = validate_gstin("99AALCS0297D1ZE")
        assert result["valid"] is False
        assert "state code" in result["reason"].lower()

    def test_invalid_pan_format_returns_invalid(self):
        result = validate_gstin("29123CS0297D1ZE")
        assert result["valid"] is False
        assert "PAN" in result["reason"]

    def test_missing_z_at_position_14_returns_invalid(self):
        result = validate_gstin("29AALCS0297D1AE")
        assert result["valid"] is False
        assert "Z" in result["reason"]

    def test_bad_checksum_returns_invalid(self):
        result = validate_gstin("29AALCS0297D1ZX")
        assert result["valid"] is False
        assert "checksum" in result["reason"].lower()

    def test_generate_valid_gstin_passes_validation(self):
        for state in ["01", "07", "27", "29", "33"]:
            gstin = generate_valid_gstin(state_code=state, pan="ABCDE1234F", entity="1")
            result = validate_gstin(gstin)
            assert result["valid"] is True, f"Generated GSTIN {gstin} failed: {result['reason']}"


class TestHSNDatabase:
    def test_database_has_entries(self):
        assert len(HSN_DATABASE) >= 50

    def test_entries_have_required_fields(self):
        for hsn_code, entry in HSN_DATABASE.items():
            assert "description" in entry, f"HSN {hsn_code} missing description"
            assert "tax_rate" in entry, f"HSN {hsn_code} missing tax_rate"
            assert "keywords" in entry, f"HSN {hsn_code} missing keywords"
            assert entry["tax_rate"] in (0, 5, 12, 18, 28), f"HSN {hsn_code} has invalid rate {entry['tax_rate']}"

    def test_hsn_codes_are_8_digits(self):
        for hsn_code in HSN_DATABASE:
            assert len(hsn_code) == 8, f"HSN code {hsn_code} is not 8 digits"
            assert hsn_code.isdigit(), f"HSN code {hsn_code} contains non-digits"


class TestTaxComputation:
    def test_intrastate_splits_cgst_sgst(self):
        result = compute_tax(
            taxable_value=10000.0,
            tax_rate=18,
            business_state="29",
            place_of_supply="29",
        )
        assert result["is_interstate"] is False
        assert result["cgst"] == 900.0
        assert result["sgst"] == 900.0
        assert result["igst"] == 0.0
        assert result["total_tax"] == 1800.0

    def test_interstate_applies_igst(self):
        result = compute_tax(
            taxable_value=10000.0,
            tax_rate=18,
            business_state="29",
            place_of_supply="27",
        )
        assert result["is_interstate"] is True
        assert result["cgst"] == 0.0
        assert result["sgst"] == 0.0
        assert result["igst"] == 1800.0
        assert result["total_tax"] == 1800.0

    def test_five_percent_rate(self):
        result = compute_tax(
            taxable_value=5000.0,
            tax_rate=5,
            business_state="07",
            place_of_supply="07",
        )
        assert result["cgst"] == 125.0
        assert result["sgst"] == 125.0
        assert result["total_tax"] == 250.0

    def test_zero_rated(self):
        result = compute_tax(
            taxable_value=5000.0,
            tax_rate=0,
            business_state="07",
            place_of_supply="07",
        )
        assert result["total_tax"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_gst_rules.py -v
```

Expected: ImportError — `server.gst_rules` does not exist yet.

- [ ] **Step 3: Implement gst_rules.py**

```python
# server/gst_rules.py
"""GST rules engine: GSTIN validation, HSN database, tax computation."""

from typing import Dict, List, Optional
import re

# ---------- GSTIN VALIDATION ----------

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

VALID_STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana",
    "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
    "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
    "16": "Tripura", "17": "Meghalaya", "18": "Assam",
    "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "25": "Daman & Diu", "26": "Dadra & Nagar Haveli",
    "27": "Maharashtra", "28": "Andhra Pradesh", "29": "Karnataka",
    "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
    "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman & Nicobar",
    "36": "Telangana", "37": "Andhra Pradesh (New)",
}


def _luhn_mod36_checksum(input_str: str) -> str:
    """Compute Luhn mod-36 checksum character."""
    n = 36
    factor = 2
    total = 0
    for char in input_str:
        code_point = GSTIN_CHARS.index(char.upper())
        addend = factor * code_point
        factor = 1 if factor == 2 else 2
        addend = (addend // n) + (addend % n)
        total += addend
    remainder = total % n
    check_code_point = (n - remainder) % n
    return GSTIN_CHARS[check_code_point]


def validate_gstin(gstin: str) -> dict:
    """Validate a GSTIN string. Returns {valid: bool, reason: str}."""
    if not isinstance(gstin, str):
        return {"valid": False, "reason": "GSTIN must be a string"}

    gstin = gstin.upper().strip()

    if len(gstin) != 15:
        return {"valid": False, "reason": "GSTIN must be exactly 15 characters"}

    state_code = gstin[:2]
    if state_code not in VALID_STATE_CODES:
        return {"valid": False, "reason": f"Invalid state code: {state_code}"}

    pan = gstin[2:12]
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan):
        return {"valid": False, "reason": f"Invalid PAN format: {pan}"}

    entity = gstin[12]
    if entity not in GSTIN_CHARS[1:]:
        return {"valid": False, "reason": f"Invalid entity code: {entity}"}

    if gstin[13] != "Z":
        return {"valid": False, "reason": "Character at position 14 must be 'Z'"}

    expected_check = _luhn_mod36_checksum(gstin[:14])
    if gstin[14] != expected_check:
        return {
            "valid": False,
            "reason": f"Checksum mismatch: expected '{expected_check}', got '{gstin[14]}'",
        }

    return {"valid": True, "reason": "Valid GSTIN"}


def generate_valid_gstin(state_code: str, pan: str, entity: str) -> str:
    """Generate a valid GSTIN with correct checksum."""
    base = f"{state_code}{pan.upper()}{entity}Z"
    checksum = _luhn_mod36_checksum(base)
    return base + checksum


# ---------- HSN DATABASE ----------

HSN_DATABASE: Dict[str, dict] = {
    # Food & Beverages (5%)
    "02011000": {"description": "Frozen bovine meat", "tax_rate": 5, "keywords": ["frozen meat", "beef", "bovine"]},
    "04011000": {"description": "Fresh milk, not concentrated", "tax_rate": 0, "keywords": ["milk", "fresh milk", "dairy"]},
    "04021000": {"description": "Milk powder", "tax_rate": 5, "keywords": ["milk powder", "powdered milk"]},
    "09011100": {"description": "Coffee, not roasted", "tax_rate": 5, "keywords": ["coffee beans", "raw coffee", "unroasted coffee"]},
    "09021000": {"description": "Green tea", "tax_rate": 5, "keywords": ["green tea", "tea leaves"]},
    "10011900": {"description": "Wheat (other than seed)", "tax_rate": 0, "keywords": ["wheat", "gehun", "grain"]},
    "10061000": {"description": "Rice in the husk (paddy)", "tax_rate": 0, "keywords": ["rice", "paddy", "chawal"]},
    "11010000": {"description": "Wheat or meslin flour", "tax_rate": 0, "keywords": ["wheat flour", "atta", "maida"]},
    "17011400": {"description": "Raw cane sugar", "tax_rate": 5, "keywords": ["sugar", "raw sugar", "cane sugar"]},
    "19011000": {"description": "Infant food preparations", "tax_rate": 5, "keywords": ["baby food", "infant formula", "cerelac"]},
    "19053100": {"description": "Sweet biscuits", "tax_rate": 18, "keywords": ["biscuits", "cookies", "sweet biscuits"]},
    "20011000": {"description": "Pickles", "tax_rate": 12, "keywords": ["pickle", "achar", "preserved vegetables"]},
    "21069099": {"description": "Namkeen and mixtures", "tax_rate": 12, "keywords": ["namkeen", "bhujia", "mixture", "snacks"]},
    "22011010": {"description": "Packaged drinking water", "tax_rate": 18, "keywords": ["mineral water", "packaged water", "bottled water"]},
    "22021010": {"description": "Aerated waters with sugar", "tax_rate": 28, "keywords": ["cola", "soft drink", "soda", "aerated beverage"]},
    # Textiles (5-12%)
    "52051100": {"description": "Cotton yarn, single, uncombed", "tax_rate": 5, "keywords": ["cotton yarn", "thread", "cotton thread"]},
    "52081100": {"description": "Plain weave cotton fabric", "tax_rate": 5, "keywords": ["cotton fabric", "cotton cloth", "plain cotton"]},
    "61091000": {"description": "T-shirts, knitted cotton", "tax_rate": 5, "keywords": ["t-shirt", "tshirt", "cotton tshirt", "knitted shirt"]},
    "62034200": {"description": "Men's trousers, cotton", "tax_rate": 12, "keywords": ["trousers", "pants", "cotton trousers", "men pants"]},
    "62042200": {"description": "Women's cotton dress", "tax_rate": 12, "keywords": ["dress", "cotton dress", "women dress", "ladies dress"]},
    "63014000": {"description": "Knitted blankets", "tax_rate": 12, "keywords": ["blanket", "knitted blanket", "bed blanket"]},
    # Chemicals & Pharma (12-18%)
    "30049099": {"description": "Medicaments, retail packs", "tax_rate": 12, "keywords": ["medicine", "tablets", "capsules", "pharmaceutical", "drugs"]},
    "33041000": {"description": "Lip makeup preparations", "tax_rate": 18, "keywords": ["lipstick", "lip gloss", "lip makeup"]},
    "33049100": {"description": "Beauty powder", "tax_rate": 18, "keywords": ["face powder", "talcum", "cosmetic powder"]},
    "33051000": {"description": "Shampoo", "tax_rate": 18, "keywords": ["shampoo", "hair wash", "hair shampoo"]},
    "33061000": {"description": "Toothpaste", "tax_rate": 18, "keywords": ["toothpaste", "dental cream", "tooth paste"]},
    "34011100": {"description": "Toilet soap", "tax_rate": 18, "keywords": ["soap", "bath soap", "toilet soap", "bathing soap"]},
    "38089100": {"description": "Insecticides", "tax_rate": 18, "keywords": ["insecticide", "pesticide", "mosquito repellent"]},
    # Plastics & Rubber (18%)
    "39231000": {"description": "Plastic boxes and cases", "tax_rate": 18, "keywords": ["plastic box", "plastic container", "plastic case", "storage box"]},
    "39269099": {"description": "Other articles of plastics", "tax_rate": 18, "keywords": ["plastic article", "plastic product", "plastic item"]},
    "40111000": {"description": "New rubber tyres for cars", "tax_rate": 28, "keywords": ["car tyre", "automobile tyre", "rubber tyre", "tire"]},
    # Iron & Steel (18%)
    "72082700": {"description": "Hot-rolled steel coils", "tax_rate": 18, "keywords": ["steel coil", "hot rolled steel", "steel sheet", "HR coil"]},
    "73089000": {"description": "Steel structures", "tax_rate": 18, "keywords": ["steel structure", "iron structure", "metal structure"]},
    "73101000": {"description": "Steel tanks and containers", "tax_rate": 18, "keywords": ["steel tank", "metal tank", "storage tank"]},
    # Machinery (18%)
    "84143000": {"description": "Compressors for refrigeration", "tax_rate": 18, "keywords": ["compressor", "refrigeration compressor", "ac compressor"]},
    "84181000": {"description": "Combined refrigerator-freezer", "tax_rate": 18, "keywords": ["refrigerator", "fridge", "freezer", "double door fridge"]},
    "84501200": {"description": "Washing machines, top load", "tax_rate": 18, "keywords": ["washing machine", "top load washer", "laundry machine"]},
    "84713000": {"description": "Portable computers (laptops)", "tax_rate": 18, "keywords": ["laptop", "notebook computer", "portable computer"]},
    "84715000": {"description": "Desktop computers", "tax_rate": 18, "keywords": ["desktop", "desktop computer", "PC", "personal computer"]},
    # Electronics (18-28%)
    "85044000": {"description": "Static converters (chargers)", "tax_rate": 18, "keywords": ["charger", "power adapter", "static converter", "usb charger"]},
    "85171200": {"description": "Mobile phones", "tax_rate": 12, "keywords": ["mobile phone", "smartphone", "cell phone", "android phone", "iphone"]},
    "85181000": {"description": "Microphones", "tax_rate": 18, "keywords": ["microphone", "mic", "wireless mic"]},
    "85183000": {"description": "Headphones and earphones", "tax_rate": 18, "keywords": ["headphone", "earphone", "earbuds", "wireless earbuds", "bluetooth earbuds"]},
    "85219000": {"description": "Video recording apparatus", "tax_rate": 18, "keywords": ["camera", "video camera", "webcam", "dash cam"]},
    "85258000": {"description": "Television cameras and CCTV", "tax_rate": 18, "keywords": ["cctv", "security camera", "surveillance camera", "ip camera"]},
    "85285100": {"description": "Computer monitors", "tax_rate": 18, "keywords": ["monitor", "computer monitor", "display", "led monitor"]},
    "85287100": {"description": "Television sets", "tax_rate": 28, "keywords": ["television", "tv", "smart tv", "led tv"]},
    # Vehicles (28%)
    "87032100": {"description": "Motor cars, petrol < 1500cc", "tax_rate": 28, "keywords": ["car", "petrol car", "small car", "hatchback"]},
    "87032200": {"description": "Motor cars, petrol 1500-3000cc", "tax_rate": 28, "keywords": ["sedan", "suv", "mid size car"]},
    "87112010": {"description": "Motorcycles 75-250cc", "tax_rate": 28, "keywords": ["motorcycle", "bike", "motorbike", "two wheeler"]},
    # Furniture (18-28%)
    "94013000": {"description": "Swivel seats with variable height", "tax_rate": 18, "keywords": ["office chair", "swivel chair", "computer chair"]},
    "94017100": {"description": "Upholstered metal frame seats", "tax_rate": 18, "keywords": ["metal chair", "dining chair", "upholstered chair"]},
    "94032000": {"description": "Metal furniture", "tax_rate": 18, "keywords": ["metal furniture", "steel almirah", "metal rack", "steel rack"]},
    "94034000": {"description": "Wooden kitchen furniture", "tax_rate": 28, "keywords": ["kitchen cabinet", "modular kitchen", "wooden cabinet"]},
    "94035000": {"description": "Wooden bedroom furniture", "tax_rate": 28, "keywords": ["bed", "wooden bed", "bedroom furniture", "wardrobe"]},
    # Services (SAC codes — 18%)
    "99831100": {"description": "Management consulting", "tax_rate": 18, "keywords": ["consulting", "advisory", "management consulting"]},
    "99831200": {"description": "IT consulting and software", "tax_rate": 18, "keywords": ["software", "it services", "software development", "web development"]},
    "99721100": {"description": "Accounting and bookkeeping", "tax_rate": 18, "keywords": ["accounting", "bookkeeping", "audit", "ca services"]},
    "99731100": {"description": "Legal services", "tax_rate": 18, "keywords": ["legal", "lawyer", "advocate", "legal services"]},
    "99611100": {"description": "Courier services", "tax_rate": 18, "keywords": ["courier", "delivery", "parcel", "shipping"]},
    "99711100": {"description": "Leasing of commercial property", "tax_rate": 18, "keywords": ["rent", "commercial rent", "office rent", "shop rent"]},
    "99851100": {"description": "Advertising services", "tax_rate": 18, "keywords": ["advertising", "ads", "marketing", "digital marketing"]},
    "99611200": {"description": "Transportation of goods", "tax_rate": 5, "keywords": ["freight", "transport", "goods transport", "trucking", "logistics"]},
}


def lookup_hsn(product_description: str) -> Optional[dict]:
    """Find the best matching HSN code for a product description.

    Returns {hsn_code, description, tax_rate} or None if no match.
    Uses keyword matching against the HSN database.
    """
    desc_lower = product_description.lower()
    best_match = None
    best_score = 0

    for hsn_code, entry in HSN_DATABASE.items():
        score = 0
        for keyword in entry["keywords"]:
            if keyword.lower() in desc_lower:
                score += len(keyword)  # longer keyword matches score higher
        if score > best_score:
            best_score = score
            best_match = {
                "hsn_code": hsn_code,
                "description": entry["description"],
                "tax_rate": entry["tax_rate"],
            }

    return best_match


# ---------- TAX COMPUTATION ----------

def compute_tax(
    taxable_value: float,
    tax_rate: int,
    business_state: str,
    place_of_supply: str,
) -> dict:
    """Compute GST breakdown for an invoice.

    If business_state == place_of_supply: intra-state (CGST + SGST, each half).
    Otherwise: inter-state (full IGST).
    """
    is_interstate = business_state != place_of_supply
    total_tax = round(taxable_value * tax_rate / 100, 2)

    if is_interstate:
        return {
            "cgst": 0.0,
            "sgst": 0.0,
            "igst": total_tax,
            "cess": 0.0,
            "total_tax": total_tax,
            "is_interstate": True,
        }
    else:
        half = round(total_tax / 2, 2)
        # Handle rounding: if half*2 != total_tax, add 0.01 to cgst
        cgst = half
        sgst = round(total_tax - half, 2)
        return {
            "cgst": cgst,
            "sgst": sgst,
            "igst": 0.0,
            "cess": 0.0,
            "total_tax": total_tax,
            "is_interstate": False,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_gst_rules.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add server/gst_rules.py tests/test_gst_rules.py
git commit -m "feat: GST rules engine — GSTIN validation, HSN database, tax computation"
```

---

### Task 3: Synthetic Data Generator

**Files:**
- Create: `gst-compliance-gym/server/data_generator.py`
- Create: `gst-compliance-gym/tests/test_data_generator.py`

- [ ] **Step 1: Write failing tests for data generator**

```python
# tests/test_data_generator.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.data_generator import generate_episode_data
from server.gst_rules import validate_gstin


class TestGenerateEpisodeData:
    def test_easy_task_returns_5_invoices(self):
        data = generate_episode_data(seed=42, task_id="easy")
        assert len(data["invoices"]) == 5

    def test_medium_task_returns_10_invoices(self):
        data = generate_episode_data(seed=42, task_id="medium")
        assert len(data["invoices"]) == 10

    def test_hard_task_returns_15_invoices(self):
        data = generate_episode_data(seed=42, task_id="hard")
        assert len(data["invoices"]) == 15

    def test_business_has_valid_gstin(self):
        data = generate_episode_data(seed=42, task_id="easy")
        result = validate_gstin(data["business"]["gstin"])
        assert result["valid"] is True

    def test_invoices_have_required_fields(self):
        data = generate_episode_data(seed=42, task_id="easy")
        required = ["invoice_id", "vendor_name", "vendor_gstin", "product_description",
                     "hsn_code", "taxable_value", "invoice_date", "invoice_number",
                     "place_of_supply", "invoice_type"]
        for inv in data["invoices"]:
            for field in required:
                assert field in inv, f"Invoice {inv.get('invoice_id')} missing {field}"

    def test_easy_task_has_2_errors(self):
        data = generate_episode_data(seed=42, task_id="easy")
        error_ids = [inv["invoice_id"] for inv in data["invoices"] if inv["invoice_id"] in data["ground_truth"]["invalid_invoices"]]
        assert len(error_ids) == 2

    def test_hard_task_has_fraud_invoices(self):
        data = generate_episode_data(seed=42, task_id="hard")
        assert len(data["ground_truth"]["fraud_invoices"]) == 2
        assert len(data["ground_truth"]["fraud_pattern"]) > 0

    def test_hard_task_has_gstr2b_data(self):
        data = generate_episode_data(seed=42, task_id="hard")
        assert "gstr2b_records" in data
        assert len(data["gstr2b_records"]) > 0

    def test_same_seed_produces_identical_data(self):
        data1 = generate_episode_data(seed=123, task_id="medium")
        data2 = generate_episode_data(seed=123, task_id="medium")
        assert data1["business"]["gstin"] == data2["business"]["gstin"]
        assert len(data1["invoices"]) == len(data2["invoices"])
        for i in range(len(data1["invoices"])):
            assert data1["invoices"][i]["invoice_id"] == data2["invoices"][i]["invoice_id"]
            assert data1["invoices"][i]["taxable_value"] == data2["invoices"][i]["taxable_value"]

    def test_different_seeds_produce_different_data(self):
        data1 = generate_episode_data(seed=1, task_id="easy")
        data2 = generate_episode_data(seed=2, task_id="easy")
        assert data1["business"]["gstin"] != data2["business"]["gstin"]

    def test_ground_truth_has_correct_tax_totals(self):
        data = generate_episode_data(seed=42, task_id="medium")
        gt = data["ground_truth"]
        assert "total_taxable_value" in gt
        assert "total_cgst" in gt
        assert "total_sgst" in gt
        assert "total_igst" in gt
        assert gt["total_taxable_value"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_data_generator.py -v
```

Expected: ImportError — `server.data_generator` does not exist yet.

- [ ] **Step 3: Implement data_generator.py**

```python
# server/data_generator.py
"""Seeded synthetic data generator for GST compliance episodes."""

import random
import string
from typing import Dict, List, Optional

try:
    from .gst_rules import (
        generate_valid_gstin,
        validate_gstin,
        HSN_DATABASE,
        compute_tax,
        VALID_STATE_CODES,
    )
except ImportError:
    from server.gst_rules import (
        generate_valid_gstin,
        validate_gstin,
        HSN_DATABASE,
        compute_tax,
        VALID_STATE_CODES,
    )


BUSINESS_NAMES = [
    "Sharma Electronics", "Patel Traders", "Gupta Textiles Pvt Ltd",
    "Singh Hardware & Tools", "Verma Enterprises", "Mehta Pharma Distributors",
    "Joshi Auto Parts", "Reddy Food Products", "Nair IT Solutions",
    "Pillai Exports", "Iyer & Sons Trading", "Das Steel Industries",
    "Banerjee Chemicals", "Mishra Agro Products", "Choudhary Plastics",
    "Agarwal Furniture House", "Kapoor Cosmetics", "Bhatia Transport Co",
    "Saxena Legal Associates", "Malhotra Advertising Agency",
]

VENDOR_NAMES = [
    "Raj Suppliers", "Krishna Enterprises", "Balaji Trading Co",
    "Shree Ganesh Distributors", "Lakshmi Agencies", "Om Sai Traders",
    "Bharat Industries", "National Wholesale Mart", "Star Exports India",
    "Royal Trading House", "Sunrise Merchants", "Golden Gate Imports",
    "Silver Line Distributors", "Diamond Traders", "Pearl Commodities",
    "Ruby Wholesale", "Emerald Supplies", "Sapphire Logistics",
    "Crystal Clear Solutions", "Platinum Services India",
]


def _generate_random_pan(rng: random.Random) -> str:
    """Generate a random PAN-like string: ABCDE1234F."""
    letters = string.ascii_uppercase
    part1 = "".join(rng.choices(letters, k=5))
    part2 = "".join(rng.choices(string.digits, k=4))
    part3 = rng.choice(letters)
    return f"{part1}{part2}{part3}"


def _corrupt_gstin(gstin: str, rng: random.Random) -> tuple:
    """Corrupt a valid GSTIN and return (corrupted, error_type)."""
    method = rng.choice(["bad_checksum", "bad_state", "bad_pan"])
    chars = list(gstin)
    if method == "bad_checksum":
        # Change last character to wrong checksum
        wrong = rng.choice([c for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != chars[14]])
        chars[14] = wrong
        return "".join(chars), "invalid_gstin"
    elif method == "bad_state":
        chars[0] = "9"
        chars[1] = "9"
        return "".join(chars), "invalid_gstin"
    else:  # bad_pan
        chars[2] = "1"  # PAN must start with letter
        chars[3] = "2"
        return "".join(chars), "invalid_gstin"


def generate_episode_data(seed: int, task_id: str) -> dict:
    """Generate a complete episode dataset.

    Args:
        seed: Random seed for reproducibility.
        task_id: One of "easy", "medium", "hard".

    Returns:
        Dictionary with business, invoices, ground_truth, and optionally gstr2b_records.
    """
    rng = random.Random(seed)

    invoice_counts = {"easy": 5, "medium": 10, "hard": 15}
    n_invoices = invoice_counts[task_id]

    # Generate business profile
    state_code = rng.choice(list(VALID_STATE_CODES.keys()))
    business_pan = _generate_random_pan(rng)
    business_gstin = generate_valid_gstin(state_code, business_pan, "1")
    business_name = rng.choice(BUSINESS_NAMES)

    business = {
        "name": business_name,
        "gstin": business_gstin,
        "state_code": state_code,
        "state_name": VALID_STATE_CODES[state_code],
        "pan": business_pan,
        "tax_period": "March 2026",
    }

    # Pick HSN codes for invoices
    hsn_codes = list(HSN_DATABASE.keys())
    invoices = []
    ground_truth_taxes = []

    for i in range(n_invoices):
        inv_id = f"INV-{i+1:03d}"
        vendor_name = rng.choice(VENDOR_NAMES)
        vendor_state = rng.choice(list(VALID_STATE_CODES.keys()))
        vendor_pan = _generate_random_pan(rng)
        vendor_gstin = generate_valid_gstin(vendor_state, vendor_pan, "1")

        hsn_code = rng.choice(hsn_codes)
        hsn_entry = HSN_DATABASE[hsn_code]
        taxable_value = round(rng.uniform(1000, 500000), 2)

        # Determine place of supply (80% same as business state)
        if rng.random() < 0.8:
            place_of_supply = state_code
        else:
            place_of_supply = rng.choice(list(VALID_STATE_CODES.keys()))

        tax_result = compute_tax(taxable_value, hsn_entry["tax_rate"], state_code, place_of_supply)

        invoice_date = f"2026-03-{rng.randint(1, 28):02d}"
        invoice_number = f"{vendor_name[:3].upper()}/{rng.randint(1000, 9999)}"

        invoice = {
            "invoice_id": inv_id,
            "vendor_name": vendor_name,
            "vendor_gstin": vendor_gstin,
            "product_description": hsn_entry["description"],
            "hsn_code": hsn_code,
            "taxable_value": taxable_value,
            "cgst": tax_result["cgst"],
            "sgst": tax_result["sgst"],
            "igst": tax_result["igst"],
            "cess": 0.0,
            "total_tax": tax_result["total_tax"],
            "total_value": round(taxable_value + tax_result["total_tax"], 2),
            "invoice_date": invoice_date,
            "invoice_number": invoice_number,
            "place_of_supply": place_of_supply,
            "place_of_supply_name": VALID_STATE_CODES[place_of_supply],
            "invoice_type": "B2B" if rng.random() < 0.7 else "B2C",
        }

        invoices.append(invoice)
        ground_truth_taxes.append(tax_result)

    # Ground truth for correct return
    total_taxable = round(sum(inv["taxable_value"] for inv in invoices), 2)
    total_cgst = round(sum(inv["cgst"] for inv in invoices), 2)
    total_sgst = round(sum(inv["sgst"] for inv in invoices), 2)
    total_igst = round(sum(inv["igst"] for inv in invoices), 2)

    ground_truth = {
        "total_taxable_value": total_taxable,
        "total_cgst": total_cgst,
        "total_sgst": total_sgst,
        "total_igst": total_igst,
        "total_cess": 0.0,
        "invalid_invoices": {},  # inv_id -> error_type
        "fraud_invoices": [],
        "fraud_pattern": "",
        "correct_hsn": {inv["invoice_id"]: inv["hsn_code"] for inv in invoices},
        "correct_taxes": {inv["invoice_id"]: ground_truth_taxes[i] for i, inv in enumerate(invoices)},
    }

    result = {
        "business": business,
        "invoices": invoices,
        "ground_truth": ground_truth,
    }

    # Inject errors based on task
    if task_id == "easy":
        _inject_easy_errors(result, rng)
    elif task_id == "medium":
        _inject_medium_errors(result, rng)
    elif task_id == "hard":
        _inject_hard_errors(result, rng)

    return result


def _inject_easy_errors(data: dict, rng: random.Random):
    """Inject 2 field-level errors into invoices."""
    invoices = data["invoices"]
    error_indices = rng.sample(range(len(invoices)), 2)

    for idx in error_indices:
        inv = invoices[idx]
        error_type = rng.choice(["invalid_gstin", "missing_fields"])

        if error_type == "invalid_gstin":
            corrupted, _ = _corrupt_gstin(inv["vendor_gstin"], rng)
            inv["vendor_gstin"] = corrupted
            data["ground_truth"]["invalid_invoices"][inv["invoice_id"]] = "invalid_gstin"
        else:
            inv["place_of_supply"] = ""
            inv["place_of_supply_name"] = ""
            data["ground_truth"]["invalid_invoices"][inv["invoice_id"]] = "missing_fields"


def _inject_medium_errors(data: dict, rng: random.Random):
    """Use ambiguous product descriptions; some invoices have wrong HSN pre-filled."""
    invoices = data["invoices"]
    # For medium, we replace product descriptions with more ambiguous versions
    # and mark 3 invoices with intentionally wrong HSN codes
    wrong_hsn_indices = rng.sample(range(len(invoices)), 3)
    hsn_codes = list(HSN_DATABASE.keys())

    for idx in wrong_hsn_indices:
        inv = invoices[idx]
        # Pick a wrong HSN code from a different chapter
        correct_chapter = inv["hsn_code"][:2]
        wrong_options = [h for h in hsn_codes if h[:2] != correct_chapter]
        if wrong_options:
            wrong_hsn = rng.choice(wrong_options)
            inv["hsn_code"] = wrong_hsn
            # Recalculate tax with wrong rate (this is what the invoice "claims")
            wrong_rate = HSN_DATABASE[wrong_hsn]["tax_rate"]
            wrong_tax = compute_tax(
                inv["taxable_value"], wrong_rate,
                data["business"]["state_code"], inv["place_of_supply"]
            )
            inv["cgst"] = wrong_tax["cgst"]
            inv["sgst"] = wrong_tax["sgst"]
            inv["igst"] = wrong_tax["igst"]
            inv["total_tax"] = wrong_tax["total_tax"]
            inv["total_value"] = round(inv["taxable_value"] + wrong_tax["total_tax"], 2)


def _inject_hard_errors(data: dict, rng: random.Random):
    """Inject ITC mismatches and circular trading fraud."""
    invoices = data["invoices"]
    business = data["business"]

    # Generate GSTR-2B records (supplier-side data for reconciliation)
    gstr2b_records = {}
    for inv in invoices:
        # Most records match perfectly
        gstr2b_records[inv["invoice_id"]] = {
            "supplier_gstin": inv["vendor_gstin"],
            "invoice_number": inv["invoice_number"],
            "invoice_date": inv["invoice_date"],
            "taxable_value": inv["taxable_value"],
            "total_tax": inv["total_tax"],
            "itc_eligible": True,
        }

    # Inject 3 ITC mismatches
    mismatch_indices = rng.sample(range(len(invoices)), min(3, len(invoices)))
    itc_mismatches = {}

    for i, idx in enumerate(mismatch_indices):
        inv = invoices[idx]
        inv_id = inv["invoice_id"]

        if i == 0:
            # Amount mismatch
            gstr2b_records[inv_id]["taxable_value"] = round(inv["taxable_value"] * 0.9, 2)
            gstr2b_records[inv_id]["total_tax"] = round(inv["total_tax"] * 0.9, 2)
            itc_mismatches[inv_id] = {
                "type": "amount_mismatch",
                "field": "taxable_value",
                "expected": inv["taxable_value"],
                "actual": gstr2b_records[inv_id]["taxable_value"],
            }
        elif i == 1:
            # Missing from supplier side
            del gstr2b_records[inv_id]
            itc_mismatches[inv_id] = {
                "type": "missing_from_supplier",
                "field": "invoice",
                "expected": "present",
                "actual": "missing",
            }
        else:
            # Duplicate claim (same invoice number, different amount)
            gstr2b_records[inv_id]["invoice_number"] = inv["invoice_number"] + "-DUP"
            itc_mismatches[inv_id] = {
                "type": "duplicate_invoice",
                "field": "invoice_number",
                "expected": inv["invoice_number"],
                "actual": gstr2b_records[inv_id]["invoice_number"],
            }

    data["gstr2b_records"] = gstr2b_records
    data["ground_truth"]["itc_mismatches"] = itc_mismatches

    # Inject circular trading fraud (2 invoices)
    # Create 3 shell entities that trade in a circle: A -> B -> C -> A
    shell_states = rng.sample(list(VALID_STATE_CODES.keys()), 3)
    shell_entities = []
    for s in shell_states:
        pan = _generate_random_pan(rng)
        gstin = generate_valid_gstin(s, pan, "1")
        shell_entities.append({"gstin": gstin, "state": s, "pan": pan})

    # Pick 2 non-mismatch invoices to mark as fraudulent
    available = [i for i in range(len(invoices)) if invoices[i]["invoice_id"] not in itc_mismatches]
    fraud_indices = rng.sample(available, min(2, len(available)))

    fraud_invoice_ids = []
    for fi, idx in enumerate(fraud_indices):
        inv = invoices[idx]
        # Make these invoices part of the circular chain
        inv["vendor_gstin"] = shell_entities[fi]["gstin"]
        inv["vendor_name"] = f"Shell Corp {fi+1}"
        fraud_invoice_ids.append(inv["invoice_id"])

        # Update GSTR-2B to match (fraud is invisible without cross-referencing)
        if inv["invoice_id"] in gstr2b_records:
            gstr2b_records[inv["invoice_id"]]["supplier_gstin"] = shell_entities[fi]["gstin"]

    data["ground_truth"]["fraud_invoices"] = fraud_invoice_ids
    data["ground_truth"]["fraud_pattern"] = (
        f"Circular trading: {shell_entities[0]['gstin']} -> "
        f"{shell_entities[1]['gstin']} -> {shell_entities[2]['gstin']} -> "
        f"{shell_entities[0]['gstin']}"
    )
    data["ground_truth"]["shell_entities"] = [e["gstin"] for e in shell_entities]
    data["ground_truth"]["invalid_invoices"].update(
        {fid: "circular_trading" for fid in fraud_invoice_ids}
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_data_generator.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add server/data_generator.py tests/test_data_generator.py
git commit -m "feat: synthetic data generator with seeded invoices and error injection"
```

---

### Task 4: Task Graders

**Files:**
- Create: `gst-compliance-gym/server/graders.py`
- Create: `gst-compliance-gym/tests/test_graders.py`

- [ ] **Step 1: Write failing tests for graders**

```python
# tests/test_graders.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.graders import grade_easy, grade_medium, grade_hard


class TestEasyGrader:
    def test_perfect_score(self):
        ground_truth = {
            "invalid_invoices": {"INV-002": "invalid_gstin", "INV-004": "missing_fields"},
        }
        agent_flags = {
            "INV-002": "invalid_gstin",
            "INV-004": "missing_fields",
        }
        score = grade_easy(agent_flags, ground_truth, total_invoices=5)
        assert score == 1.0

    def test_correct_flag_wrong_reason(self):
        ground_truth = {
            "invalid_invoices": {"INV-002": "invalid_gstin", "INV-004": "missing_fields"},
        }
        agent_flags = {
            "INV-002": "missing_fields",  # wrong reason
            "INV-004": "missing_fields",  # correct
        }
        score = grade_easy(agent_flags, ground_truth, total_invoices=5)
        # (0.10 + 0.25) / 0.50 = 0.70
        assert abs(score - 0.70) < 0.01

    def test_false_positive_penalty(self):
        ground_truth = {
            "invalid_invoices": {"INV-002": "invalid_gstin", "INV-004": "missing_fields"},
        }
        agent_flags = {
            "INV-001": "invalid_gstin",  # false positive
        }
        score = grade_easy(agent_flags, ground_truth, total_invoices=5)
        # (0 + 0 - 0.15) / 0.50 = clamped to 0.0
        assert score == 0.0

    def test_no_flags_returns_zero(self):
        ground_truth = {
            "invalid_invoices": {"INV-002": "invalid_gstin"},
        }
        score = grade_easy({}, ground_truth, total_invoices=5)
        assert score == 0.0


class TestMediumGrader:
    def test_perfect_score(self):
        ground_truth = {
            "correct_hsn": {"INV-001": "85183000", "INV-002": "84713000"},
            "correct_taxes": {
                "INV-001": {"cgst": 900.0, "sgst": 900.0, "igst": 0.0, "is_interstate": False},
                "INV-002": {"cgst": 0.0, "sgst": 0.0, "igst": 1800.0, "is_interstate": True},
            },
        }
        agent_results = {
            "INV-001": {"hsn_code": "85183000", "is_interstate": False, "total_tax": 1800.0},
            "INV-002": {"hsn_code": "84713000", "is_interstate": True, "total_tax": 1800.0},
        }
        score = grade_medium(agent_results, ground_truth)
        assert score == 1.0

    def test_partial_hsn_match(self):
        ground_truth = {
            "correct_hsn": {"INV-001": "85183000"},
            "correct_taxes": {
                "INV-001": {"cgst": 900.0, "sgst": 900.0, "igst": 0.0, "is_interstate": False},
            },
        }
        agent_results = {
            "INV-001": {"hsn_code": "85181000", "is_interstate": False, "total_tax": 1800.0},
        }
        score = grade_medium(agent_results, ground_truth)
        # 6-digit prefix match: 0.025 + interstate correct: 0.02 + tax correct: 0.04 = 0.085
        # Per invoice max = 0.1, score = 0.085/0.1 = 0.85
        assert abs(score - 0.85) < 0.01

    def test_no_results_returns_zero(self):
        ground_truth = {
            "correct_hsn": {"INV-001": "85183000"},
            "correct_taxes": {
                "INV-001": {"cgst": 900.0, "sgst": 900.0, "igst": 0.0, "is_interstate": False},
            },
        }
        score = grade_medium({}, ground_truth)
        assert score == 0.0


class TestHardGrader:
    def test_perfect_score(self):
        ground_truth = {
            "itc_mismatches": {
                "INV-003": {"type": "amount_mismatch"},
                "INV-007": {"type": "missing_from_supplier"},
                "INV-011": {"type": "duplicate_invoice"},
            },
            "fraud_invoices": ["INV-005", "INV-009"],
            "fraud_pattern": "Circular trading: A -> B -> C -> A",
        }
        agent_reconciliation = {
            "INV-003": {"match_status": "mismatched", "reason": "amount_mismatch"},
            "INV-007": {"match_status": "mismatched", "reason": "missing_from_supplier"},
            "INV-011": {"match_status": "mismatched", "reason": "duplicate_invoice"},
            "INV-001": {"match_status": "matched"},
            "INV-002": {"match_status": "matched"},
            "INV-004": {"match_status": "matched"},
        }
        agent_fraud_flags = {
            "INV-005": "circular_trading",
            "INV-009": "circular_trading",
        }
        score = grade_hard(agent_reconciliation, agent_fraud_flags, ground_truth, total_invoices=15)
        assert score == 1.0

    def test_zero_score_with_no_work(self):
        ground_truth = {
            "itc_mismatches": {"INV-003": {"type": "amount_mismatch"}},
            "fraud_invoices": ["INV-005"],
            "fraud_pattern": "Circular trading",
        }
        score = grade_hard({}, {}, ground_truth, total_invoices=15)
        assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_graders.py -v
```

Expected: ImportError — `server.graders` does not exist yet.

- [ ] **Step 3: Implement graders.py**

```python
# server/graders.py
"""Deterministic task graders for GST compliance environment."""

from typing import Dict


def grade_easy(
    agent_flags: Dict[str, str],
    ground_truth: dict,
    total_invoices: int,
) -> float:
    """Grade Task 1: Invoice Validation.

    Scoring:
        - Correct flag with correct reason: +0.25
        - Correct flag, wrong reason: +0.10
        - False positive (flagging valid invoice): -0.15
        - Missed invalid invoice: +0.0

    Returns score between 0.0 and 1.0.
    """
    invalid = ground_truth["invalid_invoices"]  # {inv_id: error_type}
    max_possible = len(invalid) * 0.25
    if max_possible == 0:
        return 1.0  # No errors to find = perfect score

    score = 0.0
    for inv_id, reason in agent_flags.items():
        if inv_id in invalid:
            if reason == invalid[inv_id]:
                score += 0.25  # Correct flag, correct reason
            else:
                score += 0.10  # Correct flag, wrong reason
        else:
            score -= 0.15  # False positive

    return max(0.0, min(1.0, score / max_possible))


def grade_medium(
    agent_results: Dict[str, dict],
    ground_truth: dict,
) -> float:
    """Grade Task 2: Tax Computation & HSN Classification.

    Per invoice (each worth 1/N of total):
        - Correct HSN 8-digit: 0.04
        - Correct HSN 6-digit prefix: 0.025
        - Correct inter/intra-state: 0.02
        - Correct tax amount (within Rs 1): 0.04

    Returns score between 0.0 and 1.0.
    """
    correct_hsn = ground_truth["correct_hsn"]
    correct_taxes = ground_truth["correct_taxes"]
    n_invoices = len(correct_hsn)
    if n_invoices == 0:
        return 1.0

    per_invoice_max = 0.1  # 0.04 + 0.02 + 0.04
    total_max = per_invoice_max * n_invoices
    total_score = 0.0

    for inv_id in correct_hsn:
        if inv_id not in agent_results:
            continue

        result = agent_results[inv_id]
        inv_score = 0.0

        # HSN matching
        agent_hsn = result.get("hsn_code", "")
        true_hsn = correct_hsn[inv_id]
        if agent_hsn == true_hsn:
            inv_score += 0.04  # Exact match
        elif len(agent_hsn) >= 6 and len(true_hsn) >= 6 and agent_hsn[:6] == true_hsn[:6]:
            inv_score += 0.025  # 6-digit prefix match
        # else: 0 points for HSN

        # Inter/intra-state
        true_tax = correct_taxes[inv_id]
        if result.get("is_interstate") == true_tax["is_interstate"]:
            inv_score += 0.02

        # Tax amount
        true_total = true_tax.get("cgst", 0) + true_tax.get("sgst", 0) + true_tax.get("igst", 0)
        agent_total = result.get("total_tax", 0)
        if abs(agent_total - true_total) <= 1.0:
            inv_score += 0.04

        total_score += inv_score

    return max(0.0, min(1.0, total_score / total_max))


def grade_hard(
    agent_reconciliation: Dict[str, dict],
    agent_fraud_flags: Dict[str, str],
    ground_truth: dict,
    total_invoices: int,
) -> float:
    """Grade Task 3: ITC Reconciliation & Fraud Detection.

    ITC Reconciliation (0.5 total):
        - Correct match/mismatch decision: +0.08 each (up to 6)
        - Correct mismatch reason: +0.04 each (up to 3)

    Fraud Detection (0.5 total):
        - Both fraudulent invoices found: +0.25
        - Only 1 found: +0.10
        - Circular pattern identified: +0.20
        - False fraud flag: -0.10

    Returns score between 0.0 and 1.0.
    """
    itc_mismatches = ground_truth.get("itc_mismatches", {})
    fraud_invoices = set(ground_truth.get("fraud_invoices", []))

    # --- ITC Reconciliation scoring (up to 0.5) ---
    itc_score = 0.0

    # Score match/mismatch decisions
    reconciled_ids = set(agent_reconciliation.keys())
    for inv_id, recon in agent_reconciliation.items():
        status = recon.get("match_status", "")
        if inv_id in itc_mismatches:
            if status == "mismatched":
                itc_score += 0.08  # Correct: it IS mismatched
                # Check reason
                agent_reason = recon.get("reason", "")
                true_reason = itc_mismatches[inv_id].get("type", "")
                if agent_reason == true_reason:
                    itc_score += 0.04
        else:
            if status == "matched":
                itc_score += 0.08  # Correct: it IS matched
            # Don't penalize wrong match decisions here (penalties in fraud section)

    itc_score = min(itc_score, 0.5)

    # --- Fraud Detection scoring (up to 0.5) ---
    fraud_score = 0.0

    flagged_fraud = set(agent_fraud_flags.keys())
    correct_fraud_flags = flagged_fraud & fraud_invoices
    false_fraud_flags = flagged_fraud - fraud_invoices

    if len(correct_fraud_flags) == len(fraud_invoices) and len(fraud_invoices) > 0:
        fraud_score += 0.25  # Found all
    elif len(correct_fraud_flags) == 1:
        fraud_score += 0.10  # Found partial

    # Check if circular pattern was identified
    for inv_id, reason in agent_fraud_flags.items():
        if inv_id in fraud_invoices and "circular" in reason.lower():
            fraud_score += 0.20
            break

    # Penalize false flags
    fraud_score -= len(false_fraud_flags) * 0.10

    fraud_score = max(0.0, min(fraud_score, 0.5))

    total = itc_score + fraud_score
    return max(0.0, min(1.0, total))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_graders.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add server/graders.py tests/test_graders.py
git commit -m "feat: deterministic graders for easy, medium, hard tasks"
```

---

### Task 5: Core Environment with MCP Tools

**Files:**
- Modify: `gst-compliance-gym/server/gst_environment.py`
- Create: `gst-compliance-gym/tests/test_environment.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_environment.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from server.gst_environment import GSTComplianceEnvironment


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = GSTComplianceEnvironment()
        obs = env.reset(seed=42, task_id="easy")
        assert obs.done is False
        assert obs.reward == 0.0
        assert "task_id" in obs.metadata
        assert obs.metadata["task_id"] == "easy"

    def test_reset_sets_state(self):
        env = GSTComplianceEnvironment()
        env.reset(seed=42, task_id="easy")
        state = env.state
        assert state.step_count == 0
        assert state.episode_id is not None

    def test_reset_with_different_tasks(self):
        env = GSTComplianceEnvironment()
        for task in ["easy", "medium", "hard"]:
            obs = env.reset(seed=42, task_id=task)
            assert obs.metadata["task_id"] == task
            assert obs.metadata["n_invoices"] in [5, 10, 15]


class TestEnvironmentTools:
    def setup_method(self):
        self.env = GSTComplianceEnvironment()
        self.env.reset(seed=42, task_id="easy")

    def test_list_tools_returns_8_tools(self):
        from openenv.core.env_server.mcp_types import ListToolsAction
        obs = self.env.step(ListToolsAction())
        tools = obs.metadata.get("tools", [])
        tool_names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        expected = ["get_invoices", "get_invoice_details", "validate_gstin",
                     "classify_hsn", "compute_tax", "reconcile_invoice",
                     "flag_invoice", "submit_return"]
        for name in expected:
            assert name in tool_names, f"Missing tool: {name}"

    def test_get_invoices_returns_list(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        obs = self.env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert obs.done is False
        result = obs.metadata.get("result", "")
        # Result should be parseable and contain invoices
        assert "INV-001" in str(result)

    def test_step_increments_count(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        self.env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert self.env.state.step_count == 1
        self.env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert self.env.state.step_count == 2

    def test_submit_return_ends_episode(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        obs = self.env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": json.dumps({
                "total_taxable_value": 0,
                "total_cgst": 0,
                "total_sgst": 0,
                "total_igst": 0,
                "total_cess": 0,
                "flagged_invoices": [],
                "itc_claimed": 0,
            })},
        ))
        assert obs.done is True
        assert isinstance(obs.reward, (int, float))
        assert 0.0 <= obs.reward <= 1.0

    def test_validate_gstin_tool(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        obs = self.env.step(CallToolAction(
            tool_name="validate_gstin",
            arguments={"gstin": "29AALCS0297D1ZX"},
        ))
        assert obs.done is False
        assert "valid" in str(obs.metadata.get("result", "")).lower() or "invalid" in str(obs.metadata.get("result", "")).lower()

    def test_max_steps_ends_episode(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        env = GSTComplianceEnvironment()
        env.reset(seed=42, task_id="easy")
        env._max_steps = 3  # Override for test
        for _ in range(3):
            obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert obs.done is True


class TestReproducibility:
    def test_same_seed_same_results(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        env1 = GSTComplianceEnvironment()
        env1.reset(seed=42, task_id="easy")
        obs1 = env1.step(CallToolAction(tool_name="get_invoices", arguments={}))

        env2 = GSTComplianceEnvironment()
        env2.reset(seed=42, task_id="easy")
        obs2 = env2.step(CallToolAction(tool_name="get_invoices", arguments={}))

        assert str(obs1.metadata.get("result")) == str(obs2.metadata.get("result"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_environment.py -v
```

Expected: Failures — skeleton environment doesn't have tools or data yet.

- [ ] **Step 3: Implement full gst_environment.py**

```python
# server/gst_environment.py
"""GST Compliance Gym — OpenEnv MCPEnvironment with 8 MCP tools."""

import json
from typing import Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, State

try:
    from .data_generator import generate_episode_data
    from .gst_rules import validate_gstin, lookup_hsn, compute_tax, HSN_DATABASE, VALID_STATE_CODES
    from .graders import grade_easy, grade_medium, grade_hard
except ImportError:
    from server.data_generator import generate_episode_data
    from server.gst_rules import validate_gstin, lookup_hsn, compute_tax, HSN_DATABASE, VALID_STATE_CODES
    from server.graders import grade_easy, grade_medium, grade_hard


class GSTComplianceEnvironment(MCPEnvironment):
    """Indian GST compliance auditing environment.

    An AI agent audits invoices for a business, validates GSTINs,
    classifies HSN codes, computes taxes, reconciles ITC claims,
    detects fraud, and submits GSTR returns.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        mcp = FastMCP("gst_compliance_gym")
        self._register_tools(mcp)
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_data = None
        self._task_id = "easy"
        self._max_steps = 50
        self._cumulative_reward = 0.0
        self._call_history = []  # Track (tool_name, args) for repeat detection
        self._agent_flags = {}  # inv_id -> reason
        self._agent_results = {}  # inv_id -> {hsn_code, is_interstate, total_tax}
        self._agent_reconciliation = {}  # inv_id -> {match_status, reason}
        self._agent_fraud_flags = {}  # inv_id -> reason
        self._invoices_viewed = set()

    def _register_tools(self, mcp: FastMCP):
        env = self  # Closure reference

        @mcp.tool
        def get_invoices() -> str:
            """List all invoices for the current tax period. Returns invoice summaries with id, vendor, amount, type, and date."""
            if env._episode_data is None:
                return json.dumps({"error": "No episode active. Call reset() first."})
            summaries = []
            for inv in env._episode_data["invoices"]:
                summaries.append({
                    "invoice_id": inv["invoice_id"],
                    "vendor_name": inv["vendor_name"],
                    "taxable_value": inv["taxable_value"],
                    "invoice_type": inv["invoice_type"],
                    "invoice_date": inv["invoice_date"],
                })
            env._step_reward = 0.01
            return json.dumps({"invoices": summaries, "count": len(summaries)})

        @mcp.tool
        def get_invoice_details(invoice_id: str) -> str:
            """Get full details of a specific invoice including all mandatory fields: GSTIN, HSN code, product description, taxable value, tax amounts, place of supply, etc."""
            if env._episode_data is None:
                return json.dumps({"error": "No episode active."})
            for inv in env._episode_data["invoices"]:
                if inv["invoice_id"] == invoice_id:
                    env._invoices_viewed.add(invoice_id)
                    env._step_reward = 0.01
                    return json.dumps(inv)
            env._step_reward = 0.0
            return json.dumps({"error": f"Invoice {invoice_id} not found."})

        @mcp.tool
        def validate_gstin_tool(gstin: str) -> str:
            """Validate a GSTIN (Goods and Services Tax Identification Number). Checks format, state code, PAN structure, and Luhn mod-36 checksum. Returns {valid: bool, reason: str}."""
            result = validate_gstin(gstin)
            if result["valid"]:
                env._step_reward = 0.02
            else:
                env._step_reward = 0.03  # Catching invalid is more valuable
            return json.dumps(result)

        @mcp.tool
        def classify_hsn(product_description: str) -> str:
            """Look up the correct HSN/SAC code for a product or service description. Returns the matching HSN code, description, and applicable GST tax rate."""
            match = lookup_hsn(product_description)
            if match:
                # Determine reward based on correctness (will be set in step)
                env._last_hsn_result = match
                env._step_reward = 0.02  # Base reward; adjusted in step if we can check
                return json.dumps(match)
            env._step_reward = 0.0
            return json.dumps({"error": "No matching HSN code found.", "suggestion": "Try a more specific product description."})

        @mcp.tool
        def compute_tax_tool(invoice_id: str, hsn_code: str, place_of_supply: str) -> str:
            """Calculate GST breakdown (CGST/SGST/IGST) for an invoice given its HSN code and place of supply. Determines if transaction is inter-state or intra-state."""
            if env._episode_data is None:
                return json.dumps({"error": "No episode active."})

            # Find invoice
            invoice = None
            for inv in env._episode_data["invoices"]:
                if inv["invoice_id"] == invoice_id:
                    invoice = inv
                    break
            if not invoice:
                env._step_reward = 0.0
                return json.dumps({"error": f"Invoice {invoice_id} not found."})

            # Look up tax rate from HSN
            if hsn_code not in HSN_DATABASE:
                env._step_reward = -0.02
                return json.dumps({"error": f"HSN code {hsn_code} not found in database."})

            tax_rate = HSN_DATABASE[hsn_code]["tax_rate"]
            business_state = env._episode_data["business"]["state_code"]

            if place_of_supply not in VALID_STATE_CODES:
                env._step_reward = -0.02
                return json.dumps({"error": f"Invalid place of supply code: {place_of_supply}"})

            result = compute_tax(invoice["taxable_value"], tax_rate, business_state, place_of_supply)
            result["invoice_id"] = invoice_id
            result["hsn_code"] = hsn_code
            result["taxable_value"] = invoice["taxable_value"]

            # Check correctness against ground truth
            gt = env._episode_data["ground_truth"]["correct_taxes"].get(invoice_id)
            if gt:
                gt_total = gt.get("cgst", 0) + gt.get("sgst", 0) + gt.get("igst", 0)
                computed_total = result["total_tax"]
                if abs(computed_total - gt_total) <= 1.0:
                    env._step_reward = 0.05
                else:
                    env._step_reward = -0.03

            # Store for grading
            env._agent_results[invoice_id] = {
                "hsn_code": hsn_code,
                "is_interstate": result["is_interstate"],
                "total_tax": result["total_tax"],
            }

            return json.dumps(result)

        @mcp.tool
        def reconcile_invoice(invoice_id: str) -> str:
            """Match an invoice against the supplier's GSTR-2B data for ITC (Input Tax Credit) eligibility. Returns match status, discrepancies if any, and ITC eligibility."""
            if env._episode_data is None:
                return json.dumps({"error": "No episode active."})

            gstr2b = env._episode_data.get("gstr2b_records", {})
            gt_mismatches = env._episode_data["ground_truth"].get("itc_mismatches", {})

            invoice = None
            for inv in env._episode_data["invoices"]:
                if inv["invoice_id"] == invoice_id:
                    invoice = inv
                    break
            if not invoice:
                env._step_reward = 0.0
                return json.dumps({"error": f"Invoice {invoice_id} not found."})

            if invoice_id not in gstr2b:
                result = {
                    "invoice_id": invoice_id,
                    "match_status": "missing",
                    "discrepancies": [{"field": "invoice", "expected": "present in GSTR-2B", "actual": "not found"}],
                    "itc_eligible": False,
                }
                is_correct = invoice_id in gt_mismatches
                env._step_reward = 0.05 if is_correct else -0.03
            else:
                record = gstr2b[invoice_id]
                discrepancies = []

                if abs(record["taxable_value"] - invoice["taxable_value"]) > 1.0:
                    discrepancies.append({
                        "field": "taxable_value",
                        "expected": invoice["taxable_value"],
                        "actual": record["taxable_value"],
                    })
                if record["invoice_number"] != invoice["invoice_number"]:
                    discrepancies.append({
                        "field": "invoice_number",
                        "expected": invoice["invoice_number"],
                        "actual": record["invoice_number"],
                    })
                if record["supplier_gstin"] != invoice["vendor_gstin"]:
                    discrepancies.append({
                        "field": "supplier_gstin",
                        "expected": invoice["vendor_gstin"],
                        "actual": record["supplier_gstin"],
                    })

                if discrepancies:
                    match_status = "mismatched"
                    itc_eligible = False
                else:
                    match_status = "matched"
                    itc_eligible = True

                result = {
                    "invoice_id": invoice_id,
                    "match_status": match_status,
                    "discrepancies": discrepancies,
                    "itc_eligible": itc_eligible,
                }

                # Check correctness
                if invoice_id in gt_mismatches:
                    is_correct = match_status == "mismatched"
                else:
                    is_correct = match_status == "matched"
                env._step_reward = 0.05 if is_correct else -0.03

            # Store for grading
            env._agent_reconciliation[invoice_id] = {
                "match_status": result["match_status"],
                "reason": gt_mismatches.get(invoice_id, {}).get("type", "") if result["match_status"] == "mismatched" else "",
            }

            return json.dumps(result)

        @mcp.tool
        def flag_invoice(invoice_id: str, reason: str) -> str:
            """Flag an invoice as erroneous or fraudulent. Valid reasons: invalid_gstin, missing_fields, wrong_hsn, tax_mismatch, itc_ineligible, circular_trading, duplicate_invoice."""
            if env._episode_data is None:
                return json.dumps({"error": "No episode active."})

            valid_reasons = ["invalid_gstin", "missing_fields", "wrong_hsn",
                             "tax_mismatch", "itc_ineligible", "circular_trading",
                             "duplicate_invoice"]
            if reason not in valid_reasons:
                env._step_reward = 0.0
                return json.dumps({"error": f"Invalid reason. Must be one of: {valid_reasons}"})

            gt_invalid = env._episode_data["ground_truth"]["invalid_invoices"]
            gt_fraud = env._episode_data["ground_truth"].get("fraud_invoices", [])

            if invoice_id in gt_invalid:
                correct_reason = gt_invalid[invoice_id]
                is_correct = True
                is_fraud = invoice_id in gt_fraud

                if is_fraud and "circular" in reason:
                    env._step_reward = 0.10
                    env._agent_fraud_flags[invoice_id] = reason
                elif reason == correct_reason:
                    env._step_reward = 0.08
                else:
                    env._step_reward = 0.04  # Right invoice, wrong reason

                env._agent_flags[invoice_id] = reason
                return json.dumps({
                    "accepted": True,
                    "correct": True,
                    "feedback": f"Invoice {invoice_id} correctly flagged.",
                })
            else:
                env._step_reward = -0.05  # False positive
                env._agent_flags[invoice_id] = reason
                if "circular" in reason:
                    env._agent_fraud_flags[invoice_id] = reason
                return json.dumps({
                    "accepted": True,
                    "correct": False,
                    "feedback": f"Invoice {invoice_id} appears to be valid. False positive.",
                })

        @mcp.tool
        def submit_return(return_data: str) -> str:
            """Submit the final GSTR return summary to end the episode. Input should be JSON with: total_taxable_value, total_cgst, total_sgst, total_igst, total_cess, flagged_invoices (list of ids), itc_claimed."""
            if env._episode_data is None:
                return json.dumps({"error": "No episode active."})

            try:
                data = json.loads(return_data) if isinstance(return_data, str) else return_data
            except (json.JSONDecodeError, TypeError):
                env._step_reward = 0.0
                return json.dumps({"error": "Invalid JSON in return_data."})

            gt = env._episode_data["ground_truth"]

            # Penalize submitting without looking at invoices
            if len(env._invoices_viewed) == 0:
                penalty = -0.2
            else:
                penalty = 0.0

            # Grade based on task
            if env._task_id == "easy":
                score = grade_easy(env._agent_flags, gt, len(env._episode_data["invoices"]))
            elif env._task_id == "medium":
                score = grade_medium(env._agent_results, gt)
            elif env._task_id == "hard":
                score = grade_hard(
                    env._agent_reconciliation,
                    env._agent_fraud_flags,
                    gt,
                    len(env._episode_data["invoices"]),
                )
            else:
                score = 0.0

            score = max(0.0, min(1.0, score + penalty))
            env._step_reward = score
            env._episode_done = True

            return json.dumps({
                "score": score,
                "breakdown": {
                    "task": env._task_id,
                    "invoices_inspected": len(env._invoices_viewed),
                    "flags_raised": len(env._agent_flags),
                    "steps_used": env._state.step_count,
                },
                "ground_truth": {
                    "total_taxable_value": gt["total_taxable_value"],
                    "total_cgst": gt["total_cgst"],
                    "total_sgst": gt["total_sgst"],
                    "total_igst": gt["total_igst"],
                    "invalid_invoices": list(gt["invalid_invoices"].keys()),
                },
            })

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._task_id = kwargs.get("task_id", "easy")
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        self._cumulative_reward = 0.0
        self._step_reward = 0.0
        self._episode_done = False
        self._call_history = []
        self._agent_flags = {}
        self._agent_results = {}
        self._agent_reconciliation = {}
        self._agent_fraud_flags = {}
        self._invoices_viewed = set()

        actual_seed = seed if seed is not None else hash(self._state.episode_id) % (2**31)
        self._episode_data = generate_episode_data(actual_seed, self._task_id)

        business = self._episode_data["business"]
        n_invoices = len(self._episode_data["invoices"])

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task_id": self._task_id,
                "business_name": business["name"],
                "business_gstin": business["gstin"],
                "business_state": business["state_name"],
                "tax_period": business["tax_period"],
                "n_invoices": n_invoices,
                "instructions": (
                    f"You are auditing {business['name']} (GSTIN: {business['gstin']}) "
                    f"in {business['state_name']} for {business['tax_period']}. "
                    f"There are {n_invoices} invoices to review. "
                    f"Use the available tools to validate, classify, compute, reconcile, "
                    f"and flag invoices, then submit the return."
                ),
            },
        )

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False, reward=0.0, metadata={"error": "Unknown action type."}
        )

    def step(self, action, timeout_s=None, **kwargs) -> Observation:
        self._step_reward = 0.0
        self._state.step_count += 1

        # Check max steps
        if self._state.step_count >= self._max_steps:
            self._episode_done = True
            # Auto-grade with current progress
            gt = self._episode_data["ground_truth"] if self._episode_data else {}
            if self._task_id == "easy":
                score = grade_easy(self._agent_flags, gt, len(self._episode_data["invoices"]))
            elif self._task_id == "medium":
                score = grade_medium(self._agent_results, gt)
            elif self._task_id == "hard":
                score = grade_hard(self._agent_reconciliation, self._agent_fraud_flags, gt, len(self._episode_data["invoices"]))
            else:
                score = 0.0
            score = max(0.0, score - 0.1)  # Timeout penalty
            return Observation(
                done=True,
                reward=score,
                metadata={"status": "max_steps_reached", "score": score},
            )

        # Detect repeated calls
        call_sig = str(action)
        if call_sig in self._call_history:
            self._step_reward = -0.01
        self._call_history.append(call_sig)

        # Delegate to MCPEnvironment (which routes to FastMCP tools)
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Override reward and done from our tracking
        reward = self._step_reward if not self._episode_done else self._step_reward
        self._cumulative_reward += reward

        return Observation(
            done=self._episode_done,
            reward=reward,
            metadata=obs.metadata,
        )

    async def step_async(self, action, timeout_s=None, **kwargs) -> Observation:
        self._step_reward = 0.0
        self._state.step_count += 1

        if self._state.step_count >= self._max_steps:
            self._episode_done = True
            gt = self._episode_data["ground_truth"] if self._episode_data else {}
            if self._task_id == "easy":
                score = grade_easy(self._agent_flags, gt, len(self._episode_data["invoices"]))
            elif self._task_id == "medium":
                score = grade_medium(self._agent_results, gt)
            elif self._task_id == "hard":
                score = grade_hard(self._agent_reconciliation, self._agent_fraud_flags, gt, len(self._episode_data["invoices"]))
            else:
                score = 0.0
            score = max(0.0, score - 0.1)
            return Observation(
                done=True, reward=score,
                metadata={"status": "max_steps_reached", "score": score},
            )

        call_sig = str(action)
        if call_sig in self._call_history:
            self._step_reward = -0.01
        self._call_history.append(call_sig)

        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        reward = self._step_reward
        self._cumulative_reward += reward

        return Observation(
            done=self._episode_done,
            reward=reward,
            metadata=obs.metadata,
        )

    @property
    def state(self) -> State:
        return self._state
```

Note: The MCP tools are registered with names like `validate_gstin_tool` and `compute_tax_tool` to avoid colliding with the imported function names. The agent sees them as `validate_gstin_tool` and `compute_tax_tool`. If you prefer `validate_gstin` as the tool name, rename the imported functions (e.g., `from .gst_rules import validate_gstin as _validate_gstin`). Adjust after testing.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/test_environment.py -v
```

Expected: All tests PASS. If import issues with openenv, install it first:
```bash
cd /Users/laptopfirst/gst-compliance-gym
uv pip install openenv-core[core] fastmcp
```

- [ ] **Step 5: Commit**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add server/gst_environment.py tests/test_environment.py
git commit -m "feat: full MCPEnvironment with 8 tools, state tracking, grading integration"
```

---

### Task 6: Inference Script

**Files:**
- Create: `gst-compliance-gym/inference.py`

- [ ] **Step 1: Create inference.py**

```python
"""
Inference Script — GST Compliance Gym
======================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
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

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 500
FALLBACK_ACTION = 'get_invoices()'

DEBUG = True

TOOL_CALL_RE = re.compile(
    r"([a-z_]+)\s*\((.*?)\)",
    re.DOTALL | re.IGNORECASE,
)

SYSTEM_PROMPT = textwrap.dedent("""
You are a GST compliance auditor AI. You are auditing an Indian business's invoices
for a tax period. Use the available tools to validate invoices, classify HSN codes,
compute taxes, reconcile ITC claims, detect fraud, and submit the final GSTR return.

Available tools:
1. get_invoices() - List all invoices for the period
2. get_invoice_details(invoice_id="INV-001") - Get full details of a specific invoice
3. validate_gstin_tool(gstin="29AALCS0297D1ZE") - Validate a GSTIN format and checksum
4. classify_hsn(product_description="wireless bluetooth earbuds") - Look up HSN code
5. compute_tax_tool(invoice_id="INV-001", hsn_code="85183000", place_of_supply="29") - Calculate GST
6. reconcile_invoice(invoice_id="INV-001") - Match against supplier GSTR-2B data
7. flag_invoice(invoice_id="INV-001", reason="invalid_gstin") - Flag erroneous/fraudulent invoice
   Valid reasons: invalid_gstin, missing_fields, wrong_hsn, tax_mismatch, itc_ineligible, circular_trading, duplicate_invoice
8. submit_return(return_data='{"total_taxable_value":0,"total_cgst":0,"total_sgst":0,"total_igst":0,"total_cess":0,"flagged_invoices":[],"itc_claimed":0}') - Submit final return (ends episode)

Respond with exactly ONE tool call per message in the format:
tool_name(arg1="value1", arg2="value2")

Do not include explanations. Just the tool call.
""").strip()


def parse_tool_call(response_text: str) -> Optional[Dict]:
    """Parse LLM response into tool name and arguments."""
    if not response_text:
        return None

    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = TOOL_CALL_RE.search(line)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            arguments = {}
            if args_str:
                # Parse key="value" patterns
                kv_re = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')
                for kv_match in kv_re.finditer(args_str):
                    arguments[kv_match.group(1)] = kv_match.group(2)

                # Also try key=value without quotes (for JSON)
                if not arguments:
                    kv_nq = re.compile(r"(\w+)\s*=\s*(.+?)(?:,\s*\w+=|$)")
                    for kv_match in kv_nq.finditer(args_str):
                        arguments[kv_match.group(1)] = kv_match.group(2).strip().strip("'\"")

            return {"tool_name": tool_name, "arguments": arguments}

    # Fallback: search whole response
    match = TOOL_CALL_RE.search(response_text)
    if match:
        return {"tool_name": match.group(1), "arguments": {}}

    return None


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    """Build the user prompt from current observation and history."""
    metadata = observation.metadata if hasattr(observation, 'metadata') else {}
    obs_text = json.dumps(metadata, indent=2, default=str) if metadata else "(no observation)"

    history_text = "\n".join(history[-6:]) if history else "None"

    return textwrap.dedent(f"""
    Step: {step}
    Current observation:
    {obs_text}

    Previous steps:
    {history_text}

    Respond with exactly one tool call.
    """).strip()


def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        return
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable not set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = GSTComplianceEnv.from_docker_image(
        image="gst-compliance-gym:latest",
    )

    scores = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Task: {task_id}")
        print(f"{'='*50}")

        history: List[str] = []

        try:
            result = env.reset(seed=42, task_id=task_id)
            observation = result.observation
            print(f"Business: {observation.metadata.get('business_name', 'Unknown')}")
            print(f"Invoices: {observation.metadata.get('n_invoices', '?')}")

            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    print(f"Episode ended at step {step-1}.")
                    break

                user_prompt = build_user_prompt(step, observation, history)

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                except Exception as exc:
                    print(f"  Model request failed ({exc}). Using fallback.")
                    response_text = FALLBACK_ACTION

                parsed = parse_tool_call(response_text)
                if parsed:
                    action = CallToolAction(
                        tool_name=parsed["tool_name"],
                        arguments=parsed["arguments"],
                    )
                    action_desc = f"{parsed['tool_name']}({parsed['arguments']})"
                else:
                    action = CallToolAction(tool_name="get_invoices", arguments={})
                    action_desc = "get_invoices() [fallback]"

                if DEBUG:
                    print(f"  Step {step}: {action_desc}")

                result = env.step(action)
                observation = result.observation
                last_reward = result.reward or 0.0

                history_line = f"Step {step}: {action_desc} -> reward {last_reward:+.3f}"
                history.append(history_line)

                if DEBUG:
                    print(f"    Reward: {last_reward:+.3f} | Done: {result.done}")

            scores[task_id] = last_reward
            print(f"Final score for {task_id}: {last_reward:.3f}")

        except Exception as exc:
            print(f"Error in task {task_id}: {exc}")
            scores[task_id] = 0.0

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for task_id, score in scores.items():
        print(f"  {task_id:8s}: {score:.3f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'average':8s}: {avg:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify inference.py syntax is valid**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -c "import ast; ast.parse(open('inference.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add inference.py
git commit -m "feat: baseline inference script with OpenAI client and tool call parsing"
```

---

### Task 7: README Documentation

**Files:**
- Create: `gst-compliance-gym/README.md`

- [ ] **Step 1: Write README.md**

```markdown
# GST Compliance Gym

An OpenEnv environment for training and evaluating AI agents on Indian GST (Goods & Services Tax) compliance auditing tasks.

## Motivation

India's GST system covers 14 million registered businesses. Compliance errors cost businesses crores in penalties, and Rs 1.79 lakh crore in fake ITC fraud was detected between FY21-25. This environment simulates real-world GST compliance tasks — invoice validation, HSN classification, tax computation, ITC reconciliation, and fraud detection — providing a challenging benchmark for AI agents.

Tax compliance is an ideal RL domain because:
- **Deterministic rewards**: Tax calculations have objectively correct answers
- **Multi-step reasoning**: Agents must validate → classify → compute → reconcile → submit
- **LLM-hard tasks**: Precise arithmetic, structured output, and graph reasoning challenge frontier models

## Environment Description

Each episode, the agent is assigned as a GST compliance auditor for an Indian business. The agent receives a set of invoices and must:
1. Inspect and validate invoices for errors
2. Classify products into correct HSN codes
3. Compute GST (CGST/SGST/IGST) based on place of supply
4. Reconcile invoices against supplier GSTR-2B data
5. Detect fraudulent patterns (circular trading)
6. Submit the final GSTR return

## Action Space (MCP Tools)

| Tool | Description |
|------|-------------|
| `get_invoices()` | List all invoices for the tax period |
| `get_invoice_details(invoice_id)` | Get full details of a specific invoice |
| `validate_gstin_tool(gstin)` | Validate GSTIN format and Luhn mod-36 checksum |
| `classify_hsn(product_description)` | Look up correct HSN/SAC code and tax rate |
| `compute_tax_tool(invoice_id, hsn_code, place_of_supply)` | Calculate GST breakdown |
| `reconcile_invoice(invoice_id)` | Match against supplier GSTR-2B data |
| `flag_invoice(invoice_id, reason)` | Flag erroneous or fraudulent invoice |
| `submit_return(return_data)` | Submit final GSTR summary (ends episode) |

## Observation Space

Each tool call returns a JSON response containing:
- Invoice data (vendor, GSTIN, HSN code, amounts, dates, place of supply)
- Validation results (valid/invalid with reasons)
- Tax computations (CGST, SGST, IGST, cess, total)
- Reconciliation status (matched/mismatched/missing with discrepancies)
- Grading feedback on flags (correct/incorrect)

Episode metadata includes: business name, GSTIN, state, tax period, number of invoices.

## Tasks

### Task 1: Invoice Validation (Easy)
- **5 invoices**, 2 with field-level errors (invalid GSTIN, missing fields)
- Score based on correctly flagging errors with right reasons
- **Expected difficulty**: LLMs score 0.6-0.8

### Task 2: Tax Computation & HSN Classification (Medium)
- **10 invoices** requiring HSN classification and tax calculation
- Score based on HSN accuracy, interstate determination, tax amount precision
- **Expected difficulty**: LLMs score 0.3-0.5

### Task 3: ITC Reconciliation & Fraud Detection (Hard)
- **15 invoices** with ITC mismatches and circular trading fraud
- Score based on reconciliation accuracy and fraud pattern detection
- **Expected difficulty**: LLMs score 0.1-0.3

## Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized execution)
- uv (package manager)

### Install
```bash
pip install openenv-core
git clone https://huggingface.co/spaces/lilbabycrypto/gst-compliance-gym
cd gst-compliance-gym
uv sync
```

### Run locally
```bash
uv run server
# Server starts at http://localhost:8000
```

### Docker
```bash
cd server
docker build -t gst-compliance-gym .
docker run -p 8000:8000 gst-compliance-gym
```

### Run inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
python inference.py
```

## Baseline Scores

| Task | Score | Model |
|------|-------|-------|
| Easy | TBD | Nemotron 3 Super |
| Medium | TBD | Nemotron 3 Super |
| Hard | TBD | Nemotron 3 Super |

(Scores will be updated after baseline evaluation)

## Reward Design

- **Dense per-step rewards**: Every tool call produces a reward signal (+0.01 to +0.10 for correct actions, -0.01 to -0.05 for errors)
- **Deterministic**: Same seed + same actions = same rewards
- **Partial credit**: HSN classification rewards at 8-digit, 6-digit, and chapter levels
- **Anti-gaming**: Repeated calls penalized, false positives penalized
- **Episode-end grading**: Final score (0.0-1.0) from deterministic task grader
```

- [ ] **Step 2: Commit**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add README.md
git commit -m "docs: README with environment description, tasks, setup instructions"
```

---

### Task 8: Local Testing & Validation

**Files:**
- No new files; testing existing code.

- [ ] **Step 1: Run all tests**

```bash
cd /Users/laptopfirst/gst-compliance-gym
python3.12 -m pytest tests/ -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 2: Start server locally and test endpoints**

```bash
cd /Users/laptopfirst/gst-compliance-gym
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 &
sleep 3

# Test health
curl -s http://localhost:8000/health

# Test reset
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Test state
curl -s http://localhost:8000/state

# Kill server
kill %1
```

Expected: `/health` returns `{"status": "healthy"}`, `/reset` returns observation JSON.

- [ ] **Step 3: Run openenv validate (local)**

```bash
cd /Users/laptopfirst/gst-compliance-gym
openenv validate
```

Expected: All local checks pass (pyproject.toml, openenv.yaml, server/app.py, Dockerfile).

- [ ] **Step 4: Run openenv validate against running server**

```bash
cd /Users/laptopfirst/gst-compliance-gym
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 &
sleep 3
openenv validate --url http://localhost:8000
kill %1
```

Expected: Runtime validation passes.

- [ ] **Step 5: Docker build and test**

```bash
cd /Users/laptopfirst/gst-compliance-gym
docker build -t gst-compliance-gym -f server/Dockerfile .
docker run -d -p 8000:8000 --name gst-gym-test gst-compliance-gym
sleep 5
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
docker stop gst-gym-test && docker rm gst-gym-test
```

Expected: Container starts, health check passes, reset returns valid observation.

- [ ] **Step 6: Commit any fixes from testing**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add -A
git commit -m "fix: address issues found during local testing and validation"
```

---

### Task 9: HF Spaces Deployment & Final Submission

**Files:**
- No new files.

- [ ] **Step 1: Login to HuggingFace CLI**

```bash
huggingface-cli login
```

- [ ] **Step 2: Deploy to HF Spaces**

```bash
cd /Users/laptopfirst/gst-compliance-gym
openenv push --repo-id lilbabycrypto/gst-compliance-gym
```

If `openenv push` doesn't work, manually create the Space:
```bash
huggingface-cli repo create gst-compliance-gym --type space --space-sdk docker
cd /Users/laptopfirst/gst-compliance-gym
git remote add hf https://huggingface.co/spaces/lilbabycrypto/gst-compliance-gym
git push hf main
```

- [ ] **Step 3: Verify deployment**

```bash
# Wait for Space to build (check HF Spaces UI)
# Then test:
curl -s https://lilbabycrypto-gst-compliance-gym.hf.space/health
curl -s -X POST https://lilbabycrypto-gst-compliance-gym.hf.space/reset -H "Content-Type: application/json" -d '{}'
```

Expected: Health returns 200, reset returns observation.

- [ ] **Step 4: Run openenv validate against deployed Space**

```bash
openenv validate --url https://lilbabycrypto-gst-compliance-gym.hf.space
```

Expected: All checks pass.

- [ ] **Step 5: Submit on Scaler dashboard**

Paste the HF Space URL on the Scaler hackathon dashboard:
`https://huggingface.co/spaces/lilbabycrypto/gst-compliance-gym`

- [ ] **Step 6: Final commit with all fixes**

```bash
cd /Users/laptopfirst/gst-compliance-gym
git add -A
git commit -m "chore: final polish for hackathon submission"
```
