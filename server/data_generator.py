"""
Synthetic Data Generator — generates seeded, reproducible episode data for GST compliance tasks.
"""

import random
import string
from typing import Optional

try:
    from .gst_rules import generate_valid_gstin, validate_gstin, HSN_DATABASE, compute_tax, VALID_STATE_CODES
except (ImportError, ModuleNotFoundError):
    from server.gst_rules import generate_valid_gstin, validate_gstin, HSN_DATABASE, compute_tax, VALID_STATE_CODES

# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

BUSINESS_NAMES = [
    "Mehta Textiles Pvt Ltd",
    "Gupta Chemicals & Pharma",
    "Sharma Electronics Ltd",
    "Patel Steel Industries",
    "Rao Trading Company",
    "Singh Agro Exports",
    "Kumar Machinery Works",
    "Jain Plastics Pvt Ltd",
    "Iyer Consulting Services",
    "Nair IT Solutions",
    "Reddy Foods & Beverages",
    "Chaudhary Automobiles",
    "Bose Furniture Makers",
    "Agarwal Jewellers",
    "Chopra Logistics Pvt Ltd",
    "Bansal Pharma Distributors",
    "Sinha Construction Ltd",
    "Verma Paper Mills",
    "Kulkarni Engineering",
    "Pillai Healthcare Products",
]

VENDOR_NAMES = [
    "Mukerjee Supplies Co",
    "Trivedi Raw Materials",
    "Dubey Hardware Traders",
    "Mishra Components Ltd",
    "Pandey Packaging Works",
    "Srivastava Metals Pvt Ltd",
    "Shukla Office Supplies",
    "Tiwari Electrical Goods",
    "Yadav Wholesale Mart",
    "Kapoor Distributors",
    "Malhotra Trading House",
    "Bajaj Enterprises",
    "Saxena Auto Parts",
    "Bhatt Chemicals Ltd",
    "Rastogi Textiles",
    "Ghosh Industrial Supplies",
    "Dey Farm Produce",
    "Majumdar Electronics",
    "Chatterjee Foods",
    "Naidu Building Materials",
]

# Shell company names for fraud injection
SHELL_COMPANY_NAMES = [
    "ABC Universal Traders",
    "XYZ Global Solutions",
    "PQR Multi Services",
    "LMN Trade Links",
    "DEF Commerce Ltd",
]

# State codes list (excludes some rarely-used codes for cleanliness)
STATE_CODE_LIST = [
    "06", "07", "08", "09", "19", "22", "23", "24", "27", "29",
    "33", "36", "37", "20", "21", "18", "32", "03",
]

# Task difficulty -> invoice count
TASK_INVOICE_COUNT = {
    "easy": 5,
    "medium": 10,
    "hard": 15,
}

# HSN entries list (derived once at module load)
_HSN_ITEMS = list(HSN_DATABASE.items())

# Group HSN codes by chapter (first 2 digits)
_HSN_BY_CHAPTER: dict = {}
for _hsn_code, _entry in _HSN_ITEMS:
    _chapter = _hsn_code[:2]
    _HSN_BY_CHAPTER.setdefault(_chapter, [])
    _HSN_BY_CHAPTER[_chapter].append(_hsn_code)

_HSN_CHAPTERS = list(_HSN_BY_CHAPTER.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_pan(rng: random.Random) -> str:
    """Generate a random valid PAN: 5 letters + 4 digits + 1 letter."""
    letters = string.ascii_uppercase
    pan = (
        "".join(rng.choices(letters, k=5))
        + "".join(rng.choices(string.digits, k=4))
        + rng.choice(letters)
    )
    return pan


def _make_gstin(rng: random.Random, state_code: str) -> str:
    """Generate a valid GSTIN for a given state."""
    pan = _random_pan(rng)
    return generate_valid_gstin(state_code, pan)


def _pick_hsn(rng: random.Random) -> tuple:
    """Pick a random HSN code and its entry. Returns (hsn_code, entry)."""
    hsn_code, entry = rng.choice(_HSN_ITEMS)
    return hsn_code, entry


def _pick_hsn_from_different_chapter(rng: random.Random, current_hsn: str) -> tuple:
    """Pick an HSN code from a different chapter than current_hsn."""
    current_chapter = current_hsn[:2]
    other_chapters = [c for c in _HSN_CHAPTERS if c != current_chapter]
    if not other_chapters:
        # Fallback: just pick any different code
        candidates = [item for item in _HSN_ITEMS if item[0] != current_hsn]
        return rng.choice(candidates)
    alt_chapter = rng.choice(other_chapters)
    alt_hsn_code = rng.choice(_HSN_BY_CHAPTER[alt_chapter])
    return alt_hsn_code, HSN_DATABASE[alt_hsn_code]


def _corrupt_gstin_checksum(gstin: str, rng: random.Random) -> str:
    """Corrupt the checksum (last char) of a GSTIN."""
    chars = list(VALID_STATE_CODES.keys())  # won't use; just change last char
    current_last = gstin[-1]
    alternatives = [c for c in string.ascii_uppercase + string.digits if c != current_last]
    new_last = rng.choice(alternatives)
    return gstin[:-1] + new_last


# ---------------------------------------------------------------------------
# Invoice generation
# ---------------------------------------------------------------------------

def _build_invoice(
    rng: random.Random,
    idx: int,
    business_state_code: str,
    vendor_name: str,
    vendor_gstin: str,
) -> dict:
    """Build a single clean invoice dict."""
    hsn_code, hsn_entry = _pick_hsn(rng)
    taxable_value = round(rng.uniform(1000, 500000), 2)

    # place_of_supply: 80% same state, 20% different
    if rng.random() < 0.80:
        place_of_supply = business_state_code
    else:
        other_states = [s for s in STATE_CODE_LIST if s != business_state_code]
        place_of_supply = rng.choice(other_states)

    tax_info = compute_tax(taxable_value, hsn_entry["tax_rate"], business_state_code, place_of_supply)
    total_value = round(taxable_value + tax_info["total_tax"], 2)

    day = rng.randint(1, 28)
    invoice_date = f"2026-03-{day:02d}"
    invoice_number = f"VEN/{rng.randint(1000, 9999)}"
    invoice_type = "B2B" if rng.random() < 0.70 else "B2C"

    return {
        "invoice_id": f"INV-{idx:03d}",
        "vendor_name": vendor_name,
        "vendor_gstin": vendor_gstin,
        "product_description": hsn_entry["description"],
        "hsn_code": hsn_code,
        "taxable_value": taxable_value,
        "cgst": tax_info["cgst"],
        "sgst": tax_info["sgst"],
        "igst": tax_info["igst"],
        "cess": tax_info["cess"],
        "total_tax": tax_info["total_tax"],
        "total_value": total_value,
        "invoice_date": invoice_date,
        "invoice_number": invoice_number,
        "place_of_supply": place_of_supply,
        "invoice_type": invoice_type,
        "_hsn_tax_rate": hsn_entry["tax_rate"],  # internal, for error injection
    }


# ---------------------------------------------------------------------------
# Error injection
# ---------------------------------------------------------------------------

def _inject_easy_errors(invoices: list, rng: random.Random, ground_truth: dict) -> None:
    """
    Corrupt exactly 2 invoices:
      - Option A: corrupt GSTIN checksum
      - Option B: clear place_of_supply (set to empty string)
    Tracks in ground_truth["invalid_invoices"].
    """
    indices = rng.sample(range(len(invoices)), 2)
    error_types = ["corrupt_gstin", "missing_place_of_supply"]
    rng.shuffle(error_types)

    for i, idx in enumerate(indices):
        inv = invoices[idx]
        err_type = error_types[i % len(error_types)]
        if err_type == "corrupt_gstin":
            inv["vendor_gstin"] = _corrupt_gstin_checksum(inv["vendor_gstin"], rng)
            ground_truth["invalid_invoices"][inv["invoice_id"]] = "invalid_gstin"
        else:
            inv["place_of_supply"] = ""
            ground_truth["invalid_invoices"][inv["invoice_id"]] = "missing_place_of_supply"


def _inject_medium_errors(
    invoices: list,
    rng: random.Random,
    ground_truth: dict,
    business_state_code: str,
) -> None:
    """
    3 invoices get wrong HSN codes from a different chapter.
    Recalculate their tax with the wrong rate.
    ground_truth["correct_hsn"] maps invoice_id -> correct hsn_code.
    ground_truth["correct_taxes"] maps invoice_id -> correct tax dict.
    """
    indices = rng.sample(range(len(invoices)), 3)

    for idx in indices:
        inv = invoices[idx]
        correct_hsn = inv["hsn_code"]
        correct_tax_rate = inv["_hsn_tax_rate"]

        # Store correct info in ground truth
        correct_tax_info = compute_tax(
            inv["taxable_value"], correct_tax_rate, business_state_code, inv["place_of_supply"]
        )
        ground_truth["correct_hsn"][inv["invoice_id"]] = correct_hsn
        ground_truth["correct_taxes"][inv["invoice_id"]] = {
            "cgst": correct_tax_info["cgst"],
            "sgst": correct_tax_info["sgst"],
            "igst": correct_tax_info["igst"],
            "cess": correct_tax_info["cess"],
            "total_tax": correct_tax_info["total_tax"],
        }

        # Inject wrong HSN from a different chapter
        wrong_hsn_code, wrong_hsn_entry = _pick_hsn_from_different_chapter(rng, correct_hsn)
        wrong_tax_info = compute_tax(
            inv["taxable_value"], wrong_hsn_entry["tax_rate"], business_state_code, inv["place_of_supply"]
        )

        inv["hsn_code"] = wrong_hsn_code
        inv["product_description"] = wrong_hsn_entry["description"]
        inv["_hsn_tax_rate"] = wrong_hsn_entry["tax_rate"]
        inv["cgst"] = wrong_tax_info["cgst"]
        inv["sgst"] = wrong_tax_info["sgst"]
        inv["igst"] = wrong_tax_info["igst"]
        inv["cess"] = wrong_tax_info["cess"]
        inv["total_tax"] = wrong_tax_info["total_tax"]
        inv["total_value"] = round(inv["taxable_value"] + wrong_tax_info["total_tax"], 2)


def _inject_hard_errors(
    invoices: list,
    rng: random.Random,
    ground_truth: dict,
    business_state_code: str,
) -> None:
    """
    1. Generate GSTR-2B records for all invoices.
    2. Inject 3 ITC mismatches: amount_mismatch, missing_from_supplier, duplicate_invoice.
    3. Inject 2 circular trading fraud invoices with shell company GSTINs.
    """
    # --- GSTR-2B records (mirror of each invoice) ---
    gstr2b_records = {}
    for inv in invoices:
        gstr2b_records[inv["invoice_id"]] = {
            "invoice_id": inv["invoice_id"],
            "vendor_gstin": inv["vendor_gstin"],
            "taxable_value": inv["taxable_value"],
            "cgst": inv["cgst"],
            "sgst": inv["sgst"],
            "igst": inv["igst"],
            "total_tax": inv["total_tax"],
            "status": "matched",
        }

    # --- ITC mismatches on 3 invoices ---
    itc_indices = rng.sample(range(len(invoices)), 3)
    mismatch_types = ["amount_mismatch", "missing_from_supplier", "duplicate_invoice"]

    for i, idx in enumerate(itc_indices):
        inv = invoices[idx]
        mtype = mismatch_types[i]
        rec = gstr2b_records[inv["invoice_id"]]

        if mtype == "amount_mismatch":
            # Supplier reported different taxable value in GSTR-2B
            delta = round(rng.uniform(100, 5000), 2)
            rec["taxable_value"] = round(inv["taxable_value"] - delta, 2)
            wrong_tax = compute_tax(
                rec["taxable_value"], inv["_hsn_tax_rate"], business_state_code, inv["place_of_supply"]
            )
            rec["cgst"] = wrong_tax["cgst"]
            rec["sgst"] = wrong_tax["sgst"]
            rec["igst"] = wrong_tax["igst"]
            rec["total_tax"] = wrong_tax["total_tax"]
            rec["status"] = "amount_mismatch"

        elif mtype == "missing_from_supplier":
            # Supplier hasn't filed GSTR-1; record is absent from GSTR-2B
            rec["status"] = "missing_from_supplier"
            rec["taxable_value"] = 0.0
            rec["cgst"] = 0.0
            rec["sgst"] = 0.0
            rec["igst"] = 0.0
            rec["total_tax"] = 0.0

        elif mtype == "duplicate_invoice":
            # Same invoice appears twice in GSTR-2B
            dup_id = inv["invoice_id"] + "_DUP"
            gstr2b_records[dup_id] = dict(rec)
            gstr2b_records[dup_id]["invoice_id"] = dup_id
            gstr2b_records[dup_id]["status"] = "duplicate_invoice"
            rec["status"] = "duplicate_invoice"

    ground_truth["gstr2b_records"] = gstr2b_records

    # --- Circular trading fraud invoices ---
    fraud_invoices = []
    shell_state = rng.choice(STATE_CODE_LIST)
    for j in range(2):
        shell_pan = _random_pan(rng)
        shell_gstin = generate_valid_gstin(shell_state, shell_pan)
        shell_name = rng.choice(SHELL_COMPANY_NAMES)

        hsn_code, hsn_entry = _pick_hsn(rng)
        taxable_value = round(rng.uniform(50000, 500000), 2)
        place_of_supply = rng.choice(STATE_CODE_LIST)
        tax_info = compute_tax(taxable_value, hsn_entry["tax_rate"], business_state_code, place_of_supply)
        total_value = round(taxable_value + tax_info["total_tax"], 2)

        day = rng.randint(1, 28)
        fraud_inv = {
            "invoice_id": f"FRAUD-{j + 1:03d}",
            "vendor_name": shell_name,
            "vendor_gstin": shell_gstin,
            "product_description": hsn_entry["description"],
            "hsn_code": hsn_code,
            "taxable_value": taxable_value,
            "cgst": tax_info["cgst"],
            "sgst": tax_info["sgst"],
            "igst": tax_info["igst"],
            "cess": tax_info["cess"],
            "total_tax": tax_info["total_tax"],
            "total_value": total_value,
            "invoice_date": f"2026-03-{day:02d}",
            "invoice_number": f"SHELL/{rng.randint(1000, 9999)}",
            "place_of_supply": place_of_supply,
            "invoice_type": "B2B",
            "_hsn_tax_rate": hsn_entry["tax_rate"],
            "fraud_flag": "circular_trading",
        }
        fraud_invoices.append(fraud_inv)

    ground_truth["fraud_invoices"] = fraud_invoices
    ground_truth["fraud_pattern"] = "circular_trading"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_episode_data(seed: int, task_id: str) -> dict:
    """
    Generate a complete, reproducible dataset for one GST compliance episode.

    Args:
        seed: Integer seed for random.Random — same seed + task_id produces identical data.
        task_id: One of "easy", "medium", "hard".

    Returns:
        {
            "business": { ... },
            "invoices": [ ... ],
            "ground_truth": { ... },
        }
    """
    rng = random.Random(seed)

    difficulty = task_id.lower()
    n_invoices = TASK_INVOICE_COUNT.get(difficulty, 5)

    # --- Business profile ---
    business_name = rng.choice(BUSINESS_NAMES)
    state_code = rng.choice(STATE_CODE_LIST)
    state_name = VALID_STATE_CODES[state_code]
    pan = _random_pan(rng)
    gstin = generate_valid_gstin(state_code, pan)

    business = {
        "company_name": business_name,
        "gstin": gstin,
        "state": state_name,
        "state_code": state_code,
        "pan": pan,
        "tax_period": "March 2026",
    }

    # --- Vendor pool for this episode ---
    vendor_names = rng.sample(VENDOR_NAMES, min(n_invoices, len(VENDOR_NAMES)))
    # If more invoices than vendor names, cycle
    while len(vendor_names) < n_invoices:
        vendor_names.append(rng.choice(VENDOR_NAMES))

    # Pre-generate vendor GSTINs
    vendor_gstins = []
    for vname in vendor_names:
        v_state = rng.choice(STATE_CODE_LIST)
        v_gstin = _make_gstin(rng, v_state)
        vendor_gstins.append(v_gstin)

    # --- Build invoices ---
    invoices = []
    for i in range(n_invoices):
        inv = _build_invoice(
            rng=rng,
            idx=i + 1,
            business_state_code=state_code,
            vendor_name=vendor_names[i],
            vendor_gstin=vendor_gstins[i],
        )
        invoices.append(inv)

    # --- Compute correct ground-truth totals (before error injection) ---
    total_taxable_value = round(sum(inv["taxable_value"] for inv in invoices), 2)
    total_cgst = round(sum(inv["cgst"] for inv in invoices), 2)
    total_sgst = round(sum(inv["sgst"] for inv in invoices), 2)
    total_igst = round(sum(inv["igst"] for inv in invoices), 2)
    total_cess = round(sum(inv["cess"] for inv in invoices), 2)

    ground_truth = {
        "total_taxable_value": total_taxable_value,
        "total_cgst": total_cgst,
        "total_sgst": total_sgst,
        "total_igst": total_igst,
        "total_cess": total_cess,
        "invalid_invoices": {},
        "fraud_invoices": [],
        "fraud_pattern": "",
        "correct_hsn": {},
        "correct_taxes": {},
    }

    # --- Error injection ---
    if difficulty == "easy":
        _inject_easy_errors(invoices, rng, ground_truth)
    elif difficulty == "medium":
        _inject_medium_errors(invoices, rng, ground_truth, state_code)
    elif difficulty == "hard":
        _inject_hard_errors(invoices, rng, ground_truth, state_code)

    # Strip internal keys from invoices before returning
    clean_invoices = []
    for inv in invoices:
        clean = {k: v for k, v in inv.items() if not k.startswith("_")}
        clean_invoices.append(clean)

    # For hard task, also strip internal keys from fraud invoices
    if difficulty == "hard" and ground_truth.get("fraud_invoices"):
        clean_fraud = []
        for inv in ground_truth["fraud_invoices"]:
            clean = {k: v for k, v in inv.items() if not k.startswith("_")}
            clean_fraud.append(clean)
        ground_truth["fraud_invoices"] = clean_fraud

    return {
        "business": business,
        "invoices": clean_invoices,
        "ground_truth": ground_truth,
    }
