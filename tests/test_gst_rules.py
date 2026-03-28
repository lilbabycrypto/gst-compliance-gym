"""
Tests for server/gst_rules.py — GSTIN validation, HSN database, and tax computation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.gst_rules import (
    validate_gstin,
    generate_valid_gstin,
    HSN_DATABASE,
    compute_tax,
    lookup_hsn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_GSTIN_STATES = [
    ("27", "AAAPZ1234A"),   # Maharashtra
    ("07", "AABCZ9876B"),   # Delhi
    ("33", "AACZZ5678C"),   # Tamil Nadu
    ("29", "AADPZ4321D"),   # Karnataka
    ("09", "AAEPZ1111E"),   # Uttar Pradesh
]

VALID_GSTIN_CASES = [
    generate_valid_gstin(state, pan) for state, pan in VALID_GSTIN_STATES
]


# ---------------------------------------------------------------------------
# GSTIN Validation — valid cases
# ---------------------------------------------------------------------------

class TestValidateGSTINValid:
    def test_generated_gstin_passes_validation(self):
        for gstin in VALID_GSTIN_CASES:
            result = validate_gstin(gstin)
            assert result["valid"] is True, f"Expected valid GSTIN, got: {result['reason']} for {gstin}"

    def test_valid_gstin_reason_message(self):
        gstin = VALID_GSTIN_CASES[0]
        result = validate_gstin(gstin)
        assert result["reason"] == "Valid GSTIN"


# ---------------------------------------------------------------------------
# GSTIN Validation — invalid cases
# ---------------------------------------------------------------------------

class TestValidateGSTINInvalid:
    def test_wrong_length_short(self):
        result = validate_gstin("27AAAPZ1234A1Z")
        assert result["valid"] is False
        assert "15 characters" in result["reason"]

    def test_wrong_length_long(self):
        result = validate_gstin("27AAAPZ1234A1ZX99")
        assert result["valid"] is False
        assert "15 characters" in result["reason"]

    def test_empty_string(self):
        result = validate_gstin("")
        assert result["valid"] is False

    def test_invalid_state_code_00(self):
        gstin = "00AAAPZ1234A1Z5"
        result = validate_gstin(gstin)
        assert result["valid"] is False
        assert "state code" in result["reason"].lower()

    def test_invalid_state_code_38(self):
        gstin = "38AAAPZ1234A1Z5"
        result = validate_gstin(gstin)
        assert result["valid"] is False
        assert "state code" in result["reason"].lower()

    def test_invalid_pan_format_lowercase(self):
        gstin = "27aaapz1234a1z5"
        result = validate_gstin(gstin)
        # When uppercased, PAN is AAAPZ1234A which might be valid — test with digits in wrong positions
        # Use a GSTIN where the PAN numeric section has letters
        result2 = validate_gstin("2712345ABCDE1Z5")
        assert result2["valid"] is False

    def test_invalid_pan_format_wrong_structure(self):
        # PAN must be 5 letters + 4 digits + 1 letter; put digit in first position
        # State=27, PAN= 1AAAZ1234A (digit at pos 0 of PAN)
        invalid = "271AAAZ1234A1Z5"
        result = validate_gstin(invalid)
        assert result["valid"] is False
        assert "PAN" in result["reason"]

    def test_invalid_entity_code_zero(self):
        # Entity code '0' is not valid (valid are 1-9, A-Z)
        state = "27"
        pan = "AAAPZ1234A"
        entity = "0"
        prefix = f"{state}{pan}{entity}Z"
        from server.gst_rules import _luhn_mod36_checksum
        checksum = _luhn_mod36_checksum(prefix)
        gstin = prefix + checksum
        result = validate_gstin(gstin)
        assert result["valid"] is False
        assert "entity code" in result["reason"].lower()

    def test_missing_z_at_position_14(self):
        # Take a valid GSTIN and change position 13 from 'Z' to 'A'
        valid_gstin = VALID_GSTIN_CASES[0]
        tampered = valid_gstin[:13] + "A" + valid_gstin[14]
        result = validate_gstin(tampered)
        assert result["valid"] is False
        assert "Z" in result["reason"] or "position 14" in result["reason"].lower()

    def test_bad_checksum(self):
        # Take a valid GSTIN and change the last character
        valid_gstin = VALID_GSTIN_CASES[0]
        # Cycle through 36 chars to find one that's different
        from server.gst_rules import GSTIN_CHARS
        last_char = valid_gstin[-1]
        bad_char = GSTIN_CHARS[(GSTIN_CHARS.index(last_char) + 1) % 36]
        tampered = valid_gstin[:-1] + bad_char
        result = validate_gstin(tampered)
        assert result["valid"] is False
        assert "checksum" in result["reason"].lower()

    def test_non_string_input(self):
        result = validate_gstin(123456789012345)
        assert result["valid"] is False

    def test_whitespace_stripped(self):
        valid_gstin = VALID_GSTIN_CASES[0]
        result = validate_gstin("  " + valid_gstin + "  ")
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# generate_valid_gstin — multiple states
# ---------------------------------------------------------------------------

class TestGenerateValidGSTIN:
    def test_generates_valid_gstin_for_all_test_states(self):
        for state_code, pan in VALID_GSTIN_STATES:
            gstin = generate_valid_gstin(state_code, pan)
            result = validate_gstin(gstin)
            assert result["valid"] is True, (
                f"generate_valid_gstin({state_code}, {pan}) produced invalid GSTIN {gstin}: {result['reason']}"
            )

    def test_generated_gstin_is_15_chars(self):
        gstin = generate_valid_gstin("27", "AAAPZ1234A")
        assert len(gstin) == 15

    def test_generated_gstin_has_correct_state_code(self):
        gstin = generate_valid_gstin("33", "AACZZ5678C")
        assert gstin[:2] == "33"

    def test_generated_gstin_position_14_is_z(self):
        gstin = generate_valid_gstin("27", "AAAPZ1234A")
        assert gstin[13] == "Z"

    def test_custom_entity_code(self):
        gstin = generate_valid_gstin("27", "AAAPZ1234A", entity="2")
        result = validate_gstin(gstin)
        assert result["valid"] is True
        assert gstin[12] == "2"

    def test_invalid_state_code_raises(self):
        import pytest
        with pytest.raises(ValueError):
            generate_valid_gstin("99", "AAAPZ1234A")

    def test_invalid_pan_raises(self):
        import pytest
        with pytest.raises(ValueError):
            generate_valid_gstin("27", "INVALID_PAN")


# ---------------------------------------------------------------------------
# HSN Database
# ---------------------------------------------------------------------------

VALID_TAX_RATES = {0, 5, 12, 18, 28}


class TestHSNDatabase:
    def test_database_has_at_least_60_entries(self):
        assert len(HSN_DATABASE) >= 60, f"HSN_DATABASE has only {len(HSN_DATABASE)} entries"

    def test_all_hsn_codes_are_8_digits(self):
        for code in HSN_DATABASE:
            assert len(code) == 8, f"HSN code '{code}' is not 8 digits"
            assert code.isdigit(), f"HSN code '{code}' contains non-digit characters"

    def test_all_entries_have_required_fields(self):
        required_fields = {"description", "tax_rate", "keywords"}
        for code, entry in HSN_DATABASE.items():
            missing = required_fields - set(entry.keys())
            assert not missing, f"HSN code {code} missing fields: {missing}"

    def test_all_tax_rates_are_valid(self):
        for code, entry in HSN_DATABASE.items():
            assert entry["tax_rate"] in VALID_TAX_RATES, (
                f"HSN code {code} has invalid tax rate {entry['tax_rate']}"
            )

    def test_all_descriptions_are_non_empty_strings(self):
        for code, entry in HSN_DATABASE.items():
            assert isinstance(entry["description"], str), f"Description for {code} is not a string"
            assert len(entry["description"]) > 0, f"Description for {code} is empty"

    def test_all_keywords_are_lists(self):
        for code, entry in HSN_DATABASE.items():
            assert isinstance(entry["keywords"], list), f"Keywords for {code} is not a list"
            assert len(entry["keywords"]) > 0, f"Keywords list for {code} is empty"

    def test_database_has_zero_rate_entries(self):
        zero_rate_entries = [c for c, e in HSN_DATABASE.items() if e["tax_rate"] == 0]
        assert len(zero_rate_entries) >= 1

    def test_database_has_five_rate_entries(self):
        five_rate_entries = [c for c, e in HSN_DATABASE.items() if e["tax_rate"] == 5]
        assert len(five_rate_entries) >= 1

    def test_database_has_twelve_rate_entries(self):
        twelve_rate_entries = [c for c, e in HSN_DATABASE.items() if e["tax_rate"] == 12]
        assert len(twelve_rate_entries) >= 1

    def test_database_has_eighteen_rate_entries(self):
        eighteen_rate_entries = [c for c, e in HSN_DATABASE.items() if e["tax_rate"] == 18]
        assert len(eighteen_rate_entries) >= 1

    def test_database_has_twentyeight_rate_entries(self):
        twentyeight_rate_entries = [c for c, e in HSN_DATABASE.items() if e["tax_rate"] == 28]
        assert len(twentyeight_rate_entries) >= 1

    def test_database_covers_food_category(self):
        # 0% food items should exist (e.g., wheat flour)
        food_zero = [c for c, e in HSN_DATABASE.items() if e["tax_rate"] == 0 and "flour" in e["description"].lower() or "wheat" in e["description"].lower()]
        assert len(food_zero) >= 1

    def test_database_covers_textiles(self):
        textile_entries = [
            c for c, e in HSN_DATABASE.items()
            if any(k in ["cotton fabric", "t-shirt", "trousers"] for k in e["keywords"])
        ]
        assert len(textile_entries) >= 1

    def test_database_covers_electronics(self):
        electronics_entries = [
            c for c, e in HSN_DATABASE.items()
            if any(k in ["mobile phone", "laptop", "monitor"] for k in e["keywords"])
        ]
        assert len(electronics_entries) >= 1

    def test_database_covers_vehicles(self):
        vehicle_entries = [
            c for c, e in HSN_DATABASE.items()
            if any(k in ["car", "motorcycle"] for k in e["keywords"])
        ]
        assert len(vehicle_entries) >= 1


# ---------------------------------------------------------------------------
# HSN Lookup
# ---------------------------------------------------------------------------

class TestLookupHSN:
    def test_lookup_rice(self):
        result = lookup_hsn("basmati chawal milled rice")
        assert result is not None
        assert result["hsn_code"] == "10063000"
        assert result["tax_rate"] == 5

    def test_lookup_mobile_phone(self):
        result = lookup_hsn("Samsung smartphone mobile phone")
        assert result is not None
        assert result["tax_rate"] == 18
        assert "mobile" in result["description"].lower() or "cellular" in result["description"].lower()

    def test_lookup_car(self):
        result = lookup_hsn("passenger car sedan motor")
        assert result is not None
        assert result["tax_rate"] == 28

    def test_lookup_laptop(self):
        result = lookup_hsn("HP laptop portable computer")
        assert result is not None
        assert result["hsn_code"] == "84713000"
        assert result["tax_rate"] == 18

    def test_lookup_cotton_tshirt(self):
        result = lookup_hsn("cotton t-shirt")
        assert result is not None
        assert result["tax_rate"] == 5

    def test_lookup_sugar(self):
        result = lookup_hsn("cane sugar cheeni")
        assert result is not None
        assert result["hsn_code"] == "17011200"
        assert result["tax_rate"] == 5

    def test_lookup_tyre(self):
        result = lookup_hsn("car tyre rubber pneumatic")
        assert result is not None
        assert result["tax_rate"] == 28

    def test_lookup_software_development(self):
        result = lookup_hsn("software development IT services")
        assert result is not None
        assert result["tax_rate"] == 18

    def test_lookup_returns_none_for_unknown(self):
        result = lookup_hsn("xyzzy frobnicator 99999")
        assert result is None

    def test_lookup_returns_none_for_empty(self):
        result = lookup_hsn("")
        assert result is None

    def test_lookup_result_has_required_keys(self):
        result = lookup_hsn("laptop")
        assert result is not None
        assert "hsn_code" in result
        assert "description" in result
        assert "tax_rate" in result

    def test_lookup_beer(self):
        result = lookup_hsn("beer malt lager")
        assert result is not None
        assert result["hsn_code"] == "22030000"
        assert result["tax_rate"] == 28

    def test_lookup_medicine_tablet(self):
        result = lookup_hsn("medicine tablet capsule")
        assert result is not None
        assert result["tax_rate"] == 12

    def test_lookup_washing_machine(self):
        result = lookup_hsn("fully automatic washing machine")
        assert result is not None
        assert result["hsn_code"] == "84501100"
        assert result["tax_rate"] == 18


# ---------------------------------------------------------------------------
# Tax Computation
# ---------------------------------------------------------------------------

class TestComputeTax:
    # ---- Intra-state (CGST + SGST) ----

    def test_intrastate_18_percent(self):
        result = compute_tax(10000.0, 18, "Maharashtra", "Maharashtra")
        assert result["is_interstate"] is False
        assert result["igst"] == 0.0
        assert result["cgst"] == 900.0
        assert result["sgst"] == 900.0
        assert result["total_tax"] == 1800.0

    def test_intrastate_splits_cgst_sgst_equally(self):
        result = compute_tax(1000.0, 12, "Delhi", "Delhi")
        assert result["cgst"] == result["sgst"]
        assert result["cgst"] == 60.0
        assert result["total_tax"] == 120.0

    def test_intrastate_5_percent(self):
        result = compute_tax(500.0, 5, "Karnataka", "Karnataka")
        assert result["is_interstate"] is False
        assert result["cgst"] == 12.5
        assert result["sgst"] == 12.5
        assert result["total_tax"] == 25.0
        assert result["igst"] == 0.0

    def test_intrastate_28_percent(self):
        result = compute_tax(2000.0, 28, "Tamil Nadu", "Tamil Nadu")
        assert result["is_interstate"] is False
        assert result["cgst"] == 280.0
        assert result["sgst"] == 280.0
        assert result["total_tax"] == 560.0

    # ---- Inter-state (IGST) ----

    def test_interstate_18_percent(self):
        result = compute_tax(10000.0, 18, "Maharashtra", "Delhi")
        assert result["is_interstate"] is True
        assert result["igst"] == 1800.0
        assert result["cgst"] == 0.0
        assert result["sgst"] == 0.0
        assert result["total_tax"] == 1800.0

    def test_interstate_12_percent(self):
        result = compute_tax(5000.0, 12, "Karnataka", "Tamil Nadu")
        assert result["is_interstate"] is True
        assert result["igst"] == 600.0
        assert result["cgst"] == 0.0
        assert result["sgst"] == 0.0
        assert result["total_tax"] == 600.0

    def test_interstate_5_percent(self):
        result = compute_tax(1000.0, 5, "Gujarat", "Rajasthan")
        assert result["is_interstate"] is True
        assert result["igst"] == 50.0
        assert result["total_tax"] == 50.0

    def test_interstate_28_percent(self):
        result = compute_tax(50000.0, 28, "Haryana", "Punjab")
        assert result["is_interstate"] is True
        assert result["igst"] == 14000.0
        assert result["total_tax"] == 14000.0

    # ---- Zero-rated ----

    def test_zero_rate_intrastate(self):
        result = compute_tax(10000.0, 0, "Maharashtra", "Maharashtra")
        assert result["total_tax"] == 0.0
        assert result["cgst"] == 0.0
        assert result["sgst"] == 0.0
        assert result["igst"] == 0.0

    def test_zero_rate_interstate(self):
        result = compute_tax(10000.0, 0, "Maharashtra", "Delhi")
        assert result["total_tax"] == 0.0
        assert result["igst"] == 0.0

    # ---- Cess ----

    def test_cess_is_always_zero(self):
        result = compute_tax(1000.0, 18, "Maharashtra", "Delhi")
        assert result["cess"] == 0.0

    # ---- Rounding ----

    def test_rounding_to_two_decimal_places(self):
        # 333.33 * 18% = 59.9994 → rounds to 60.0
        result = compute_tax(333.33, 18, "Maharashtra", "Delhi")
        assert isinstance(result["total_tax"], float)
        # Verify at most 2 decimal places
        assert round(result["total_tax"], 2) == result["total_tax"]

    def test_intrastate_rounding_small_amount(self):
        # 100 * 5% = 5.00 → cgst=2.50, sgst=2.50
        result = compute_tax(100.0, 5, "Delhi", "Delhi")
        assert result["cgst"] == 2.5
        assert result["sgst"] == 2.5

    # ---- Return structure ----

    def test_result_contains_all_keys(self):
        result = compute_tax(1000.0, 18, "Maharashtra", "Delhi")
        required_keys = {"cgst", "sgst", "igst", "cess", "total_tax", "is_interstate"}
        assert required_keys.issubset(set(result.keys()))

    def test_state_comparison_is_case_insensitive(self):
        result_lower = compute_tax(1000.0, 18, "maharashtra", "maharashtra")
        result_upper = compute_tax(1000.0, 18, "MAHARASHTRA", "MAHARASHTRA")
        assert result_lower["is_interstate"] is False
        assert result_upper["is_interstate"] is False

    def test_state_comparison_strips_whitespace(self):
        result = compute_tax(1000.0, 18, "  Maharashtra  ", "  Maharashtra  ")
        assert result["is_interstate"] is False
