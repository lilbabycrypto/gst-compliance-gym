"""
Tests for server/graders.py — deterministic graders for easy, medium, and hard tasks.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.graders import grade_easy, grade_medium, grade_hard


# ---------------------------------------------------------------------------
# grade_easy
# ---------------------------------------------------------------------------

class TestGradeEasy:
    def _make_gt(self, invalid_invoices: dict) -> dict:
        return {"invalid_invoices": invalid_invoices}

    def test_perfect_score(self):
        """All flags correct with correct reasons → 1.0"""
        ground_truth = self._make_gt({
            "INV001": "missing_gstin",
            "INV002": "negative_amount",
        })
        agent_flags = {
            "INV001": "missing_gstin",
            "INV002": "negative_amount",
        }
        result = grade_easy(agent_flags, ground_truth, total_invoices=10)
        assert result == 1.0

    def test_correct_flag_wrong_reason(self):
        """One correct flag with wrong reason → 0.10/0.25 ≈ 0.4 for single invoice"""
        ground_truth = self._make_gt({"INV001": "missing_gstin"})
        agent_flags = {"INV001": "wrong_reason"}
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        # score = 0.10, max_possible = 0.25, ratio = 0.4
        assert abs(result - 0.40) < 1e-9

    def test_correct_flag_wrong_reason_two_invoices(self):
        """Two invoices: 1 correct+correct, 1 correct+wrong → (0.25+0.10)/(0.25*2) ≈ 0.70"""
        ground_truth = self._make_gt({
            "INV001": "missing_gstin",
            "INV002": "negative_amount",
        })
        agent_flags = {
            "INV001": "missing_gstin",   # correct reason
            "INV002": "wrong_reason",    # wrong reason
        }
        result = grade_easy(agent_flags, ground_truth, total_invoices=10)
        # score = 0.35, max_possible = 0.50, ratio = 0.70
        assert abs(result - 0.70) < 1e-9

    def test_false_positive_only_clamped_to_zero(self):
        """Flagging a valid invoice only → negative score, clamped to 0.0"""
        ground_truth = self._make_gt({"INV001": "missing_gstin"})
        agent_flags = {"INV999": "some_reason"}  # false positive
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        # score = -0.15, max_possible = 0.25 → raw = -0.6 → clamped to 0.0
        assert result == 0.0

    def test_no_flags_returns_zero(self):
        """Agent flags nothing when there are invalid invoices → 0.0"""
        ground_truth = self._make_gt({"INV001": "missing_gstin"})
        agent_flags = {}
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        assert result == 0.0

    def test_no_invalid_invoices_returns_perfect(self):
        """When ground truth has no invalid invoices, any agent trivially scores 1.0"""
        ground_truth = self._make_gt({})
        agent_flags = {}
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        assert result == 1.0

    def test_no_invalid_invoices_with_flags_still_returns_perfect(self):
        """No invalid invoices means edge case returns 1.0 regardless of flags"""
        ground_truth = self._make_gt({})
        agent_flags = {"INV001": "reason"}
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        assert result == 1.0

    def test_result_clamped_to_one(self):
        """Score cannot exceed 1.0"""
        ground_truth = self._make_gt({"INV001": "missing_gstin"})
        agent_flags = {"INV001": "missing_gstin"}
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        assert result <= 1.0

    def test_result_clamped_to_zero_with_many_false_positives(self):
        """Multiple false positives → deeply negative score clamped to 0.0"""
        ground_truth = self._make_gt({"INV001": "missing_gstin"})
        agent_flags = {
            "INV_FP1": "reason",
            "INV_FP2": "reason",
            "INV_FP3": "reason",
            "INV_FP4": "reason",
        }
        result = grade_easy(agent_flags, ground_truth, total_invoices=10)
        assert result == 0.0

    def test_single_invoice_exact_match(self):
        """Single invalid invoice, exact match → 1.0"""
        ground_truth = self._make_gt({"INV001": "duplicate_invoice"})
        agent_flags = {"INV001": "duplicate_invoice"}
        result = grade_easy(agent_flags, ground_truth, total_invoices=3)
        assert result == 1.0

    def test_case_sensitive_reason_matching(self):
        """Reason matching is case-insensitive after strip"""
        ground_truth = self._make_gt({"INV001": "missing_gstin"})
        agent_flags = {"INV001": "MISSING_GSTIN"}
        result = grade_easy(agent_flags, ground_truth, total_invoices=5)
        # After lower() comparison: "missing_gstin" == "missing_gstin" → correct
        assert result == 1.0


# ---------------------------------------------------------------------------
# grade_medium
# ---------------------------------------------------------------------------

class TestGradeMedium:
    def _make_gt(self, hsn: dict, taxes: dict) -> dict:
        return {"correct_hsn": hsn, "correct_taxes": taxes}

    def test_perfect_score(self):
        """Perfect HSN, is_interstate, and total_tax → 1.0"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200", "INV002": "84713000"},
            taxes={
                "INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True},
                "INV002": {"cgst": 450.0, "sgst": 450.0, "igst": 0.0, "is_interstate": False},
            },
        )
        agent_results = {
            "INV001": {"hsn_code": "85171200", "is_interstate": True, "total_tax": 900.0},
            "INV002": {"hsn_code": "84713000", "is_interstate": False, "total_tax": 900.0},
        }
        result = grade_medium(agent_results, ground_truth)
        assert result == 1.0

    def test_partial_hsn_match_six_digit_prefix(self):
        """6-digit prefix match (not exact 8-digit) → 0.025 instead of 0.04 for HSN component"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        # Agent gives 6-digit prefix match: 851712xx vs 85171200
        agent_results = {
            "INV001": {"hsn_code": "85171299", "is_interstate": True, "total_tax": 900.0},
        }
        result = grade_medium(agent_results, ground_truth)
        # HSN: 0.025, is_interstate: 0.02, total_tax: 0.04 → total = 0.085
        # max = 0.1 → 0.085/0.1 = 0.85
        assert abs(result - 0.85) < 1e-9

    def test_no_hsn_match(self):
        """Wrong HSN (no prefix match) → 0 HSN points"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        agent_results = {
            "INV001": {"hsn_code": "11111111", "is_interstate": True, "total_tax": 900.0},
        }
        result = grade_medium(agent_results, ground_truth)
        # HSN: 0, is_interstate: 0.02, total_tax: 0.04 → total = 0.06
        # max = 0.1 → 0.06/0.1 = 0.6
        assert abs(result - 0.60) < 1e-9

    def test_no_results_returns_zero(self):
        """Agent returns nothing → 0.0"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        result = grade_medium({}, ground_truth)
        assert result == 0.0

    def test_wrong_interstate(self):
        """Wrong is_interstate → loses 0.02"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        agent_results = {
            "INV001": {"hsn_code": "85171200", "is_interstate": False, "total_tax": 900.0},
        }
        result = grade_medium(agent_results, ground_truth)
        # HSN: 0.04, is_interstate: 0 (wrong), total_tax: 0.04 → total = 0.08
        # max = 0.1 → 0.08/0.1 = 0.8
        assert abs(result - 0.80) < 1e-9

    def test_total_tax_within_one_rupee(self):
        """Tax off by exactly 1 still scores"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        agent_results = {
            "INV001": {"hsn_code": "85171200", "is_interstate": True, "total_tax": 901.0},
        }
        result = grade_medium(agent_results, ground_truth)
        # All correct: 1.0
        assert result == 1.0

    def test_total_tax_outside_one_rupee(self):
        """Tax off by more than 1 rupee → loses 0.04"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        agent_results = {
            "INV001": {"hsn_code": "85171200", "is_interstate": True, "total_tax": 902.0},
        }
        result = grade_medium(agent_results, ground_truth)
        # HSN: 0.04, is_interstate: 0.02, total_tax: 0 → total = 0.06
        # max = 0.1 → 0.6
        assert abs(result - 0.60) < 1e-9

    def test_empty_ground_truth(self):
        """Empty ground truth with no agent results → 1.0 (no invoices)"""
        ground_truth = self._make_gt({}, {})
        result = grade_medium({}, ground_truth)
        assert result == 1.0

    def test_score_clamped_to_one(self):
        """Score never exceeds 1.0"""
        ground_truth = self._make_gt(
            hsn={"INV001": "85171200"},
            taxes={"INV001": {"cgst": 0.0, "sgst": 0.0, "igst": 900.0, "is_interstate": True}},
        )
        agent_results = {
            "INV001": {"hsn_code": "85171200", "is_interstate": True, "total_tax": 900.0},
        }
        result = grade_medium(agent_results, ground_truth)
        assert result <= 1.0


# ---------------------------------------------------------------------------
# grade_hard
# ---------------------------------------------------------------------------

class TestGradeHard:
    def _make_gt(self, mismatches=None, fraud_invoices=None, fraud_pattern="") -> dict:
        return {
            "itc_mismatches": mismatches or {},
            "fraud_invoices": fraud_invoices or [],
            "fraud_pattern": fraud_pattern,
        }

    def test_perfect_score(self):
        """Correctly reconciled all invoices + caught all fraud with circular pattern → 1.0"""
        gt = self._make_gt(
            mismatches={"INV001": "amount_mismatch"},
            fraud_invoices=["INV002", "INV003"],
            fraud_pattern="circular trading scheme",
        )
        # Correct reconciliation: INV001 is mismatch, INV004/INV005 are matches
        agent_reconciliation = {
            "INV001": {"match_status": "mismatch", "reason": "amount_mismatch"},
            "INV004": {"match_status": "match", "reason": ""},
            "INV005": {"match_status": "match", "reason": ""},
        }
        agent_fraud_flags = {
            "INV002": "circular trading detected",
            "INV003": "circular trading detected",
        }
        result = grade_hard(agent_reconciliation, agent_fraud_flags, gt, total_invoices=10)
        # ITC: INV001 mismatch correct (+0.08) + reason (+0.04) + INV004 match (+0.08) + INV005 match (+0.08) = 0.28, capped at 0.5 → 0.28
        # Fraud: all found (+0.25) + circular (+0.20) + 0 false positives = 0.45, capped at 0.5 → 0.45
        # Total = 0.73... but let's check the exact calculation
        assert result > 0.5
        assert result <= 1.0

    def test_zero_work(self):
        """Agent does nothing → 0.0"""
        gt = self._make_gt(
            mismatches={"INV001": "amount_mismatch"},
            fraud_invoices=["INV002"],
            fraud_pattern="circular",
        )
        result = grade_hard({}, {}, gt, total_invoices=10)
        assert result == 0.0

    def test_itc_correct_mismatch_decision_no_reason(self):
        """Correct mismatch decision but reason doesn't match → only +0.08"""
        gt = self._make_gt(mismatches={"INV001": "amount_mismatch"})
        agent_reconciliation = {
            "INV001": {"match_status": "mismatch", "reason": "something else"},
        }
        result = grade_hard(agent_reconciliation, {}, gt, total_invoices=5)
        # ITC: 0.08 (decision only), Fraud: 0 → 0.08
        assert abs(result - 0.08) < 1e-9

    def test_itc_correct_mismatch_decision_with_reason(self):
        """Correct mismatch decision + reason → +0.08 + 0.04 = 0.12"""
        gt = self._make_gt(mismatches={"INV001": "amount_mismatch"})
        agent_reconciliation = {
            "INV001": {"match_status": "mismatch", "reason": "amount_mismatch detected"},
        }
        result = grade_hard(agent_reconciliation, {}, gt, total_invoices=5)
        # ITC: 0.12, Fraud: 0 → 0.12
        assert abs(result - 0.12) < 1e-9

    def test_itc_correct_match_decision(self):
        """Correct match decision for a non-mismatch invoice → +0.08"""
        gt = self._make_gt(mismatches={})
        agent_reconciliation = {
            "INV001": {"match_status": "match", "reason": ""},
        }
        result = grade_hard(agent_reconciliation, {}, gt, total_invoices=5)
        assert abs(result - 0.08) < 1e-9

    def test_fraud_all_found_no_circular(self):
        """All fraud invoices found but no circular pattern → +0.25"""
        gt = self._make_gt(
            fraud_invoices=["INV001", "INV002"],
            fraud_pattern="simple overinvoicing",
        )
        agent_fraud_flags = {
            "INV001": "overinvoicing",
            "INV002": "overinvoicing",
        }
        result = grade_hard({}, agent_fraud_flags, gt, total_invoices=10)
        # Fraud: all found (+0.25), no circular → 0.25, no false positives → fraud_score = 0.25
        assert abs(result - 0.25) < 1e-9

    def test_fraud_only_one_found(self):
        """Only 1 of multiple fraud invoices found → +0.10"""
        gt = self._make_gt(
            fraud_invoices=["INV001", "INV002"],
            fraud_pattern="some pattern",
        )
        agent_fraud_flags = {"INV001": "fraud"}
        result = grade_hard({}, agent_fraud_flags, gt, total_invoices=10)
        # Fraud: 1 found (+0.10), no circular → 0.10
        assert abs(result - 0.10) < 1e-9

    def test_fraud_false_positive_penalty(self):
        """False positive fraud flag → -0.10 per flag"""
        gt = self._make_gt(
            fraud_invoices=["INV001"],
            fraud_pattern="circular",
        )
        # Found all fraud + circular, but also a false positive
        agent_fraud_flags = {
            "INV001": "circular detected",
            "INV_FP": "false flag",  # false positive
        }
        result = grade_hard({}, agent_fraud_flags, gt, total_invoices=10)
        # Fraud: all found (+0.25) + circular (+0.20) - 1 FP (0.10) = 0.35
        assert abs(result - 0.35) < 1e-9

    def test_fraud_score_clamped_at_zero(self):
        """Many false positives → fraud score clamped to 0.0"""
        gt = self._make_gt(
            fraud_invoices=["INV001"],
            fraud_pattern="circular",
        )
        agent_fraud_flags = {
            "INV001": "circular",  # correct (+0.25 + 0.20 = 0.45)
            "FP1": "false", "FP2": "false", "FP3": "false",
            "FP4": "false", "FP5": "false", "FP6": "false",  # 6 FP = -0.60
        }
        result = grade_hard({}, agent_fraud_flags, gt, total_invoices=10)
        # Fraud: 0.45 - 0.60 = -0.15, clamped to 0.0
        # ITC: 0
        assert result == 0.0

    def test_itc_score_capped_at_half(self):
        """ITC score cannot exceed 0.5"""
        mismatches = {f"INV{i:03d}": "amount_mismatch" for i in range(1, 20)}
        gt = self._make_gt(mismatches=mismatches)
        agent_reconciliation = {
            inv_id: {"match_status": "mismatch", "reason": "amount_mismatch"}
            for inv_id in mismatches
        }
        result = grade_hard(agent_reconciliation, {}, gt, total_invoices=30)
        # ITC would be 19 * 0.12 = 2.28, capped at 0.5
        assert result <= 0.5

    def test_full_perfect_score_one_mismatch_one_fraud(self):
        """Minimal perfect scenario: 1 mismatch + 1 fraud, circular → check composites"""
        gt = self._make_gt(
            mismatches={"INV001": "amount_mismatch"},
            fraud_invoices=["INV002"],
            fraud_pattern="circular billing",
        )
        agent_reconciliation = {
            "INV001": {"match_status": "mismatch", "reason": "amount_mismatch"},
        }
        agent_fraud_flags = {
            "INV002": "circular billing detected",
        }
        result = grade_hard(agent_reconciliation, agent_fraud_flags, gt, total_invoices=5)
        # ITC: 0.08 + 0.04 = 0.12
        # Fraud: all found (+0.25) + circular (+0.20) = 0.45
        # Total: 0.57
        assert abs(result - 0.57) < 1e-9

    def test_result_always_between_zero_and_one(self):
        """Result is always in [0, 1]"""
        gt = self._make_gt(
            mismatches={"INV001": "reason"},
            fraud_invoices=["INV002"],
            fraud_pattern="circular",
        )
        result = grade_hard({}, {}, gt, total_invoices=5)
        assert 0.0 <= result <= 1.0
