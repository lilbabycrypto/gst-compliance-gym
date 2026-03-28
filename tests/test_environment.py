"""
Tests for server/gst_environment.py — GSTComplianceEnvironment with 8 MCP tools.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.gst_environment import GSTComplianceEnvironment, MAX_STEPS
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed=42, task_id="easy"):
    """Create an environment and reset it."""
    env = GSTComplianceEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    return env, obs


# ---------------------------------------------------------------------------
# Test: reset() returns valid observation
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation_not_done(self):
        env, obs = _make_env()
        assert obs.done is False

    def test_reset_returns_zero_reward(self):
        env, obs = _make_env()
        assert obs.reward == 0.0

    def test_reset_metadata_has_task_id(self):
        env, obs = _make_env()
        assert obs.metadata["task_id"] == "easy"

    def test_reset_metadata_has_business(self):
        env, obs = _make_env()
        assert "business" in obs.metadata
        assert "company_name" in obs.metadata["business"]
        assert "gstin" in obs.metadata["business"]

    def test_reset_metadata_has_invoice_count(self):
        env, obs = _make_env()
        assert "invoice_count" in obs.metadata
        assert obs.metadata["invoice_count"] > 0

    def test_reset_metadata_has_instructions(self):
        env, obs = _make_env()
        assert "instructions" in obs.metadata
        assert len(obs.metadata["instructions"]) > 0

    def test_reset_sets_step_count_to_zero(self):
        env, obs = _make_env()
        assert env.state.step_count == 0

    def test_reset_easy(self):
        env, obs = _make_env(seed=42, task_id="easy")
        assert obs.metadata["task_id"] == "easy"
        assert obs.metadata["invoice_count"] == 5

    def test_reset_medium(self):
        env, obs = _make_env(seed=42, task_id="medium")
        assert obs.metadata["task_id"] == "medium"
        assert obs.metadata["invoice_count"] == 10

    def test_reset_hard(self):
        env, obs = _make_env(seed=42, task_id="hard")
        assert obs.metadata["task_id"] == "hard"
        assert obs.metadata["invoice_count"] == 15


# ---------------------------------------------------------------------------
# Test: list_tools returns 8 tools
# ---------------------------------------------------------------------------

class TestListTools:
    def test_list_tools_returns_8_tools(self):
        env, _ = _make_env()
        obs = env.step(ListToolsAction())
        assert len(obs.tools) == 8

    def test_list_tools_expected_names(self):
        env, _ = _make_env()
        obs = env.step(ListToolsAction())
        names = {t.name for t in obs.tools}
        expected = {
            "get_invoices",
            "get_invoice_details",
            "validate_gstin_tool",
            "classify_hsn",
            "compute_tax_tool",
            "reconcile_invoice",
            "flag_invoice",
            "submit_return",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Test: get_invoices
# ---------------------------------------------------------------------------

class TestGetInvoices:
    def test_get_invoices_returns_list(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result = json.loads(obs.result.content[0].text)
        assert "invoices" in result
        assert isinstance(result["invoices"], list)

    def test_get_invoices_contains_inv001(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result = json.loads(obs.result.content[0].text)
        ids = [inv["invoice_id"] for inv in result["invoices"]]
        assert "INV-001" in ids

    def test_get_invoices_reward(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert obs.reward == 0.01


# ---------------------------------------------------------------------------
# Test: get_invoice_details
# ---------------------------------------------------------------------------

class TestGetInvoiceDetails:
    def test_get_invoice_details_returns_data(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="get_invoice_details",
            arguments={"invoice_id": "INV-001"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "vendor_name" in result
        assert "taxable_value" in result
        assert result["invoice_id"] == "INV-001"

    def test_get_invoice_details_not_found(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="get_invoice_details",
            arguments={"invoice_id": "INV-999"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "error" in result


# ---------------------------------------------------------------------------
# Test: validate_gstin_tool
# ---------------------------------------------------------------------------

class TestValidateGSTIN:
    def test_validate_valid_gstin(self):
        """The business GSTIN from the episode should be valid."""
        env, reset_obs = _make_env()
        gstin = reset_obs.metadata["business"]["gstin"]
        obs = env.step(CallToolAction(
            tool_name="validate_gstin_tool",
            arguments={"gstin": gstin},
        ))
        result = json.loads(obs.result.content[0].text)
        assert result["valid"] is True
        assert obs.reward == 0.02

    def test_validate_invalid_gstin(self):
        obs_env = GSTComplianceEnvironment()
        obs_env.reset(seed=42, task_id="easy")
        obs = obs_env.step(CallToolAction(
            tool_name="validate_gstin_tool",
            arguments={"gstin": "99ZZZZZ9999Z9Z9"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert result["valid"] is False
        assert obs.reward == 0.03


# ---------------------------------------------------------------------------
# Test: classify_hsn
# ---------------------------------------------------------------------------

class TestClassifyHSN:
    def test_classify_known_product(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="classify_hsn",
            arguments={"product_description": "laptop notebook computer"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "hsn_code" in result
        assert obs.reward == 0.05

    def test_classify_unknown_product(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="classify_hsn",
            arguments={"product_description": "xyzzyplugh quantum widget"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "error" in result
        assert obs.reward == -0.02


# ---------------------------------------------------------------------------
# Test: compute_tax_tool
# ---------------------------------------------------------------------------

class TestComputeTax:
    def test_compute_tax_valid(self):
        env, _ = _make_env()
        # First get invoice details to know HSN and place_of_supply
        obs = env.step(CallToolAction(
            tool_name="get_invoice_details",
            arguments={"invoice_id": "INV-001"},
        ))
        inv = json.loads(obs.result.content[0].text)

        obs = env.step(CallToolAction(
            tool_name="compute_tax_tool",
            arguments={
                "invoice_id": "INV-001",
                "hsn_code": inv["hsn_code"],
                "place_of_supply": inv["place_of_supply"],
            },
        ))
        result = json.loads(obs.result.content[0].text)
        assert "total_tax" in result
        assert obs.reward == 0.05

    def test_compute_tax_invalid_invoice(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="compute_tax_tool",
            arguments={
                "invoice_id": "INV-999",
                "hsn_code": "84713000",
                "place_of_supply": "27",
            },
        ))
        result = json.loads(obs.result.content[0].text)
        assert "error" in result
        assert obs.reward == -0.03


# ---------------------------------------------------------------------------
# Test: step increments step_count
# ---------------------------------------------------------------------------

class TestStepCount:
    def test_step_increments(self):
        env, _ = _make_env()
        assert env.state.step_count == 0
        env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert env.state.step_count == 1
        env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert env.state.step_count == 2


# ---------------------------------------------------------------------------
# Test: submit_return ends episode
# ---------------------------------------------------------------------------

class TestSubmitReturn:
    def test_submit_return_ends_episode(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": "{}"},
        ))
        assert obs.done is True

    def test_submit_return_score_range(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": "{}"},
        ))
        # Score is the reward (0.0 to 1.0)
        assert 0.0 <= obs.reward <= 1.0

    def test_submit_return_result_has_score(self):
        env, _ = _make_env()
        obs = env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": "{}"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "score" in result
        assert "status" in result
        assert result["status"] == "graded"

    def test_submit_after_flagging_easy(self):
        """Flag the correct invalid invoices, then submit for a non-zero score."""
        env, _ = _make_env(seed=100, task_id="easy")

        # Get the ground truth invalid invoices by inspecting episode data
        gt = env._episode_data["ground_truth"]
        invalid_invs = gt.get("invalid_invoices", {})

        # Flag each invalid invoice with the correct reason
        for inv_id, reason in invalid_invs.items():
            env.step(CallToolAction(
                tool_name="flag_invoice",
                arguments={"invoice_id": inv_id, "reason": reason},
            ))

        obs = env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": "{}"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert result["score"] > 0.0
        assert obs.done is True


# ---------------------------------------------------------------------------
# Test: max steps ends episode
# ---------------------------------------------------------------------------

class TestMaxSteps:
    def test_max_steps_terminates(self):
        env, _ = _make_env()
        # Exhaust MAX_STEPS
        for i in range(MAX_STEPS):
            obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
            if obs.done:
                break

        # The next step should exceed max steps
        obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert obs.done is True
        assert obs.reward == -0.1


# ---------------------------------------------------------------------------
# Test: reproducibility (same seed → same data)
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_invoices(self):
        env1, obs1 = _make_env(seed=42, task_id="easy")
        inv_obs1 = env1.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result1 = json.loads(inv_obs1.result.content[0].text)

        env2, obs2 = _make_env(seed=42, task_id="easy")
        inv_obs2 = env2.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result2 = json.loads(inv_obs2.result.content[0].text)

        assert result1 == result2

    def test_same_seed_same_business(self):
        _, obs1 = _make_env(seed=42, task_id="easy")
        _, obs2 = _make_env(seed=42, task_id="easy")
        assert obs1.metadata["business"] == obs2.metadata["business"]

    def test_different_seed_different_data(self):
        env1, _ = _make_env(seed=42, task_id="easy")
        inv_obs1 = env1.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result1 = json.loads(inv_obs1.result.content[0].text)

        env2, _ = _make_env(seed=99, task_id="easy")
        inv_obs2 = env2.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result2 = json.loads(inv_obs2.result.content[0].text)

        # Very unlikely to be identical with different seeds
        assert result1 != result2


# ---------------------------------------------------------------------------
# Test: repeat call penalty
# ---------------------------------------------------------------------------

class TestRepeatPenalty:
    def test_repeat_call_penalty(self):
        env, _ = _make_env()
        obs1 = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert obs1.reward == 0.01

        # Same call again → penalty
        obs2 = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        assert obs2.reward == 0.0  # 0.01 - 0.01 = 0.0


# ---------------------------------------------------------------------------
# Test: reconcile_invoice (hard mode)
# ---------------------------------------------------------------------------

class TestReconcileInvoice:
    def test_reconcile_no_gstr2b_in_easy(self):
        env, _ = _make_env(seed=42, task_id="easy")
        obs = env.step(CallToolAction(
            tool_name="reconcile_invoice",
            arguments={"invoice_id": "INV-001"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "error" in result

    def test_reconcile_in_hard_mode(self):
        env, _ = _make_env(seed=42, task_id="hard")
        obs = env.step(CallToolAction(
            tool_name="reconcile_invoice",
            arguments={"invoice_id": "INV-001"},
        ))
        result = json.loads(obs.result.content[0].text)
        assert "match_status" in result
        assert "invoice_id" in result


# ---------------------------------------------------------------------------
# Test: flag_invoice
# ---------------------------------------------------------------------------

class TestFlagInvoice:
    def test_flag_valid_invoice_penalty(self):
        """Flagging a valid invoice should have negative reward."""
        env, _ = _make_env(seed=42, task_id="easy")

        # Find an invoice that is NOT in invalid_invoices
        gt = env._episode_data["ground_truth"]
        invalid_ids = set(gt.get("invalid_invoices", {}).keys())
        all_ids = [inv["invoice_id"] for inv in env._episode_data["invoices"]]
        valid_id = None
        for iid in all_ids:
            if iid not in invalid_ids:
                valid_id = iid
                break

        if valid_id:
            obs = env.step(CallToolAction(
                tool_name="flag_invoice",
                arguments={"invoice_id": valid_id, "reason": "test"},
            ))
            assert obs.reward == -0.05

    def test_flag_invalid_invoice_reward(self):
        """Flagging an invalid invoice should have positive reward."""
        env, _ = _make_env(seed=42, task_id="easy")
        gt = env._episode_data["ground_truth"]
        invalid_ids = list(gt.get("invalid_invoices", {}).keys())
        if invalid_ids:
            obs = env.step(CallToolAction(
                tool_name="flag_invoice",
                arguments={"invoice_id": invalid_ids[0], "reason": "test"},
            ))
            assert obs.reward == 0.08


# ---------------------------------------------------------------------------
# Test: medium and hard task flows
# ---------------------------------------------------------------------------

class TestMediumTask:
    def test_medium_compute_and_submit(self):
        env, obs = _make_env(seed=42, task_id="medium")
        assert obs.metadata["task_id"] == "medium"

        # Submit return (even without doing work, should grade)
        sub_obs = env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": "{}"},
        ))
        assert sub_obs.done is True
        result = json.loads(sub_obs.result.content[0].text)
        assert "score" in result


class TestHardTask:
    def test_hard_reconcile_and_submit(self):
        env, obs = _make_env(seed=42, task_id="hard")
        assert obs.metadata["task_id"] == "hard"

        # Submit return
        sub_obs = env.step(CallToolAction(
            tool_name="submit_return",
            arguments={"return_data": "{}"},
        ))
        assert sub_obs.done is True
        result = json.loads(sub_obs.result.content[0].text)
        assert "score" in result

    def test_hard_has_fraud_invoices_in_listing(self):
        env, _ = _make_env(seed=42, task_id="hard")
        obs = env.step(CallToolAction(tool_name="get_invoices", arguments={}))
        result = json.loads(obs.result.content[0].text)
        ids = [inv["invoice_id"] for inv in result["invoices"]]
        # Hard mode should have FRAUD-001 and FRAUD-002
        assert "FRAUD-001" in ids
        assert "FRAUD-002" in ids
