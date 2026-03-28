"""
Deterministic graders for GST Compliance Gym tasks.
"""

from typing import Any


def grade_easy(agent_flags: dict, ground_truth: dict, total_invoices: int) -> float:
    """
    Grade Task 1: Invoice Validation.

    Args:
        agent_flags: dict {inv_id: reason} — invoices the agent flagged as invalid
        ground_truth: dict with key 'invalid_invoices' → {inv_id: error_type}
        total_invoices: total number of invoices in the task (unused in scoring formula,
                        kept for interface consistency)

    Returns:
        Normalised score in [0.0, 1.0].
    """
    invalid_invoices: dict = ground_truth.get("invalid_invoices", {})

    # Edge case: no invalid invoices exist — perfect score by default
    if not invalid_invoices:
        return 1.0

    max_possible = len(invalid_invoices) * 0.25

    score = 0.0
    for inv_id, reason in agent_flags.items():
        if inv_id in invalid_invoices:
            # Correct flag — check reason
            expected_error = invalid_invoices[inv_id]
            if isinstance(reason, str) and isinstance(expected_error, str):
                reason_match = reason.strip().lower() == expected_error.strip().lower()
            else:
                reason_match = reason == expected_error
            if reason_match:
                score += 0.25  # Correct flag, correct reason
            else:
                score += 0.10  # Correct flag, wrong reason
        else:
            score -= 0.15  # False positive

    # Missed invalids contribute 0 — no action needed

    return max(0.0, min(1.0, score / max_possible))


def grade_medium(agent_results: dict, ground_truth: dict) -> float:
    """
    Grade Task 2: Tax Computation & HSN Classification.

    Args:
        agent_results: dict {inv_id: {hsn_code, is_interstate, total_tax}}
        ground_truth: dict with:
            'correct_hsn'   → {inv_id: hsn_code}
            'correct_taxes' → {inv_id: {cgst, sgst, igst, is_interstate}}

    Returns:
        Normalised score in [0.0, 1.0].
    """
    correct_hsn: dict = ground_truth.get("correct_hsn", {})
    correct_taxes: dict = ground_truth.get("correct_taxes", {})

    # Determine invoice set from ground truth
    all_inv_ids = set(correct_hsn.keys()) | set(correct_taxes.keys())
    n_invoices = len(all_inv_ids)

    if n_invoices == 0:
        return 1.0 if not agent_results else 0.0

    total_max = 0.1 * n_invoices
    total_score = 0.0

    for inv_id in all_inv_ids:
        agent = agent_results.get(inv_id)
        if agent is None:
            continue  # No result for this invoice → 0 points

        inv_score = 0.0

        # HSN classification (max 0.04)
        if inv_id in correct_hsn:
            expected_hsn = str(correct_hsn[inv_id])
            agent_hsn = str(agent.get("hsn_code", ""))
            if agent_hsn == expected_hsn:
                inv_score += 0.04  # Exact 8-digit match
            elif len(agent_hsn) >= 6 and len(expected_hsn) >= 6 and agent_hsn[:6] == expected_hsn[:6]:
                inv_score += 0.025  # 6-digit prefix match (not exact)

        # is_interstate (max 0.02)
        if inv_id in correct_taxes:
            expected_taxes = correct_taxes[inv_id]
            expected_interstate = expected_taxes.get("is_interstate")
            agent_interstate = agent.get("is_interstate")
            if expected_interstate is not None and agent_interstate == expected_interstate:
                inv_score += 0.02

            # total_tax within Rs 1 (max 0.04)
            expected_total = (
                expected_taxes.get("igst", 0.0)
                + expected_taxes.get("cgst", 0.0)
                + expected_taxes.get("sgst", 0.0)
            )
            agent_total = agent.get("total_tax")
            if agent_total is not None:
                try:
                    if abs(float(agent_total) - float(expected_total)) <= 1.0:
                        inv_score += 0.04
                except (TypeError, ValueError):
                    pass

        total_score += inv_score

    return max(0.0, min(1.0, total_score / total_max))


def grade_hard(
    agent_reconciliation: dict,
    agent_fraud_flags: dict,
    ground_truth: dict,
    total_invoices: int,
) -> float:
    """
    Grade Task 3: ITC Reconciliation & Fraud Detection.

    Args:
        agent_reconciliation: dict {inv_id: {match_status, reason}}
        agent_fraud_flags: dict {inv_id: reason}
        ground_truth: dict with:
            'itc_mismatches'  → {inv_id: reason}
            'fraud_invoices'  → list/set of inv_ids
            'fraud_pattern'   → str describing the fraud pattern
        total_invoices: total number of invoices (unused in formula, kept for interface)

    Returns:
        Normalised score in [0.0, 1.0].
    """
    itc_mismatches: dict = ground_truth.get("itc_mismatches", {})
    fraud_invoices = set(ground_truth.get("fraud_invoices", []))
    fraud_pattern: str = ground_truth.get("fraud_pattern", "")

    # -----------------------------------------------------------------------
    # ITC Reconciliation score (cap at 0.5)
    # -----------------------------------------------------------------------
    itc_score = 0.0

    for inv_id, agent_rec in agent_reconciliation.items():
        agent_status = agent_rec.get("match_status", "")
        agent_reason = agent_rec.get("reason", "")

        is_mismatch = inv_id in itc_mismatches

        if is_mismatch:
            # Agent correctly identified a mismatch
            if str(agent_status).lower() in ("mismatch", "false", "no"):
                itc_score += 0.08  # Correct decision
                # Check if reason matches
                expected_reason = str(itc_mismatches[inv_id]).strip().lower()
                if expected_reason and expected_reason in str(agent_reason).strip().lower():
                    itc_score += 0.04  # Correct reason
        else:
            # Agent correctly identified a match
            if str(agent_status).lower() in ("match", "true", "yes", "ok"):
                itc_score += 0.08  # Correct decision

    itc_score = min(0.5, itc_score)

    # -----------------------------------------------------------------------
    # Fraud Detection score (cap at 0.5)
    # -----------------------------------------------------------------------
    fraud_score = 0.0

    if fraud_invoices:
        flagged_set = set(agent_fraud_flags.keys())
        correctly_found = flagged_set & fraud_invoices
        false_positives = flagged_set - fraud_invoices

        # All fraudulent invoices found
        if correctly_found == fraud_invoices:
            fraud_score += 0.25
        elif len(correctly_found) == 1:
            fraud_score += 0.10

        # Circular pattern identified
        if fraud_pattern and "circular" in fraud_pattern.lower():
            # Check if any agent flag reason mentions "circular"
            for reason in agent_fraud_flags.values():
                if "circular" in str(reason).lower():
                    fraud_score += 0.20
                    break

        # Penalise false positives
        fraud_score -= len(false_positives) * 0.10

    fraud_score = max(0.0, min(0.5, fraud_score))

    return max(0.0, min(1.0, itc_score + fraud_score))
