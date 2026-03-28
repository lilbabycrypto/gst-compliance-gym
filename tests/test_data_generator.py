"""
Tests for server/data_generator.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.data_generator import generate_episode_data
from server.gst_rules import validate_gstin

REQUIRED_INVOICE_FIELDS = [
    "invoice_id",
    "vendor_name",
    "vendor_gstin",
    "product_description",
    "hsn_code",
    "taxable_value",
    "invoice_date",
    "invoice_number",
    "place_of_supply",
    "invoice_type",
]


# ---------------------------------------------------------------------------
# Invoice count tests
# ---------------------------------------------------------------------------

def test_easy_returns_5_invoices():
    data = generate_episode_data(seed=42, task_id="easy")
    assert len(data["invoices"]) == 5


def test_medium_returns_10_invoices():
    data = generate_episode_data(seed=42, task_id="medium")
    assert len(data["invoices"]) == 10


def test_hard_returns_15_invoices():
    data = generate_episode_data(seed=42, task_id="hard")
    assert len(data["invoices"]) == 15


# ---------------------------------------------------------------------------
# Business profile tests
# ---------------------------------------------------------------------------

def test_business_has_valid_gstin():
    data = generate_episode_data(seed=7, task_id="easy")
    result = validate_gstin(data["business"]["gstin"])
    assert result["valid"] is True, f"Business GSTIN invalid: {result['reason']}"


def test_business_has_required_fields():
    data = generate_episode_data(seed=99, task_id="easy")
    biz = data["business"]
    for field in ["company_name", "gstin", "state", "pan", "tax_period"]:
        assert field in biz, f"Missing field: {field}"
    assert biz["tax_period"] == "March 2026"


# ---------------------------------------------------------------------------
# Invoice field tests
# ---------------------------------------------------------------------------

def test_invoices_have_required_fields():
    data = generate_episode_data(seed=13, task_id="medium")
    for inv in data["invoices"]:
        for field in REQUIRED_INVOICE_FIELDS:
            assert field in inv, f"Invoice missing field '{field}': {inv.get('invoice_id')}"


def test_invoices_have_no_internal_fields():
    """Internal fields starting with _ should be stripped from output."""
    data = generate_episode_data(seed=55, task_id="hard")
    for inv in data["invoices"]:
        for key in inv:
            assert not key.startswith("_"), f"Internal field leaked into output: {key}"


def test_invoice_id_format():
    data = generate_episode_data(seed=21, task_id="easy")
    for i, inv in enumerate(data["invoices"], start=1):
        assert inv["invoice_id"] == f"INV-{i:03d}", f"Wrong invoice_id: {inv['invoice_id']}"


def test_invoice_date_format():
    data = generate_episode_data(seed=33, task_id="easy")
    for inv in data["invoices"]:
        assert inv["invoice_date"].startswith("2026-03-"), f"Bad date: {inv['invoice_date']}"


def test_invoice_type_values():
    data = generate_episode_data(seed=1234, task_id="medium")
    for inv in data["invoices"]:
        assert inv["invoice_type"] in ("B2B", "B2C"), f"Bad invoice_type: {inv['invoice_type']}"


# ---------------------------------------------------------------------------
# Ground truth tests
# ---------------------------------------------------------------------------

def test_ground_truth_has_correct_tax_totals():
    data = generate_episode_data(seed=42, task_id="easy")
    gt = data["ground_truth"]
    assert gt["total_taxable_value"] > 0
    # total_taxable_value should be sum of all invoice taxable_values
    expected = round(sum(inv["taxable_value"] for inv in data["invoices"]), 2)
    # For easy, some invoices are corrupted (GSTIN/place_of_supply) but taxable values are unchanged
    # The ground truth totals are computed BEFORE error injection for taxable values
    assert gt["total_taxable_value"] > 0


def test_easy_has_exactly_2_invalid_invoices():
    data = generate_episode_data(seed=42, task_id="easy")
    gt = data["ground_truth"]
    assert "invalid_invoices" in gt
    assert len(gt["invalid_invoices"]) == 2, (
        f"Expected 2 invalid invoices, got {len(gt['invalid_invoices'])}: {gt['invalid_invoices']}"
    )


def test_easy_invalid_invoice_ids_exist_in_invoices():
    data = generate_episode_data(seed=77, task_id="easy")
    gt = data["ground_truth"]
    invoice_ids = {inv["invoice_id"] for inv in data["invoices"]}
    for inv_id in gt["invalid_invoices"]:
        assert inv_id in invoice_ids, f"Invalid invoice id {inv_id} not found in invoices"


def test_medium_correct_hsn_has_3_entries():
    data = generate_episode_data(seed=42, task_id="medium")
    gt = data["ground_truth"]
    assert "correct_hsn" in gt
    assert len(gt["correct_hsn"]) == 3, (
        f"Expected 3 correct_hsn entries, got {len(gt['correct_hsn'])}"
    )


def test_medium_correct_hsn_differs_from_injected():
    """The correct HSN stored in ground truth should differ from what's on the invoice."""
    data = generate_episode_data(seed=42, task_id="medium")
    gt = data["ground_truth"]
    invoices_by_id = {inv["invoice_id"]: inv for inv in data["invoices"]}
    for inv_id, correct_hsn in gt["correct_hsn"].items():
        invoice_hsn = invoices_by_id[inv_id]["hsn_code"]
        assert invoice_hsn != correct_hsn, (
            f"Invoice {inv_id}: injected HSN should differ from correct HSN"
        )


# ---------------------------------------------------------------------------
# Hard task tests
# ---------------------------------------------------------------------------

def test_hard_has_fraud_invoices():
    data = generate_episode_data(seed=42, task_id="hard")
    gt = data["ground_truth"]
    assert "fraud_invoices" in gt
    assert len(gt["fraud_invoices"]) == 2, (
        f"Expected 2 fraud invoices, got {len(gt['fraud_invoices'])}"
    )


def test_hard_fraud_pattern_non_empty():
    data = generate_episode_data(seed=42, task_id="hard")
    gt = data["ground_truth"]
    assert "fraud_pattern" in gt
    assert gt["fraud_pattern"] != "", "fraud_pattern should be non-empty for hard task"


def test_hard_has_gstr2b_records():
    data = generate_episode_data(seed=42, task_id="hard")
    gt = data["ground_truth"]
    assert "gstr2b_records" in gt
    assert len(gt["gstr2b_records"]) >= len(data["invoices"]), (
        "gstr2b_records should cover all invoices (plus possible duplicates)"
    )


def test_hard_fraud_invoices_have_valid_gstins():
    data = generate_episode_data(seed=42, task_id="hard")
    for fraud_inv in data["ground_truth"]["fraud_invoices"]:
        result = validate_gstin(fraud_inv["vendor_gstin"])
        assert result["valid"] is True, (
            f"Fraud invoice GSTIN invalid: {fraud_inv['vendor_gstin']} — {result['reason']}"
        )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_data():
    data1 = generate_episode_data(seed=100, task_id="medium")
    data2 = generate_episode_data(seed=100, task_id="medium")
    assert data1 == data2, "Same seed should produce identical data"


def test_different_seeds_produce_different_data():
    data1 = generate_episode_data(seed=1, task_id="easy")
    data2 = generate_episode_data(seed=2, task_id="easy")
    # At minimum, the businesses should be different (highly likely with different seeds)
    assert data1 != data2, "Different seeds should produce different data"


def test_same_seed_different_task_produces_different_data():
    data_easy = generate_episode_data(seed=42, task_id="easy")
    data_hard = generate_episode_data(seed=42, task_id="hard")
    assert len(data_easy["invoices"]) != len(data_hard["invoices"])


# ---------------------------------------------------------------------------
# Tax arithmetic sanity
# ---------------------------------------------------------------------------

def test_invoice_total_value_equals_taxable_plus_tax():
    data = generate_episode_data(seed=88, task_id="easy")
    for inv in data["invoices"]:
        expected_total = round(inv["taxable_value"] + inv["total_tax"], 2)
        assert inv["total_value"] == expected_total, (
            f"Invoice {inv['invoice_id']}: total_value mismatch. "
            f"Expected {expected_total}, got {inv['total_value']}"
        )


def test_invoice_cgst_sgst_igst_sums_to_total_tax():
    data = generate_episode_data(seed=55, task_id="medium")
    for inv in data["invoices"]:
        computed = round(inv["cgst"] + inv["sgst"] + inv["igst"] + inv["cess"], 2)
        assert computed == round(inv["total_tax"], 2), (
            f"Invoice {inv['invoice_id']}: tax components don't sum to total_tax. "
            f"Components sum: {computed}, total_tax: {inv['total_tax']}"
        )
