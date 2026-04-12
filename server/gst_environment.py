"""
GST Compliance Gym — Core MCP Environment.

Implements an MCPEnvironment subclass with 8 MCP tools that an LLM agent
calls to audit GST invoices.  The environment generates synthetic episode
data, tracks agent actions, computes step-level rewards, and uses
deterministic graders to produce a final score at episode end.
"""

import json
from typing import Optional, Any, Dict
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, State
from pydantic import Field


class GSTObservation(Observation):
    """Extended observation that includes GST task context as real fields."""
    task_id: Optional[str] = Field(default=None)
    business: Optional[Dict[str, Any]] = Field(default=None)
    invoice_count: Optional[int] = Field(default=None)
    instructions: Optional[str] = Field(default=None)
    step_count: Optional[int] = Field(default=None)
    last_tool_result: Optional[Any] = Field(default=None)

try:
    from .data_generator import generate_episode_data
    from .gst_rules import validate_gstin, lookup_hsn, compute_tax, HSN_DATABASE, VALID_STATE_CODES
    from .graders import grade_easy, grade_medium, grade_hard
except (ImportError, ModuleNotFoundError):
    from server.data_generator import generate_episode_data
    from server.gst_rules import validate_gstin, lookup_hsn, compute_tax, HSN_DATABASE, VALID_STATE_CODES
    from server.graders import grade_easy, grade_medium, grade_hard


MAX_STEPS = 50


class GSTComplianceEnvironment(MCPEnvironment):
    """Indian GST compliance auditing environment with 8 MCP tools."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        mcp = FastMCP("gst_compliance_gym")

        # ----------------------------------------------------------------
        # Capture self for closures
        # ----------------------------------------------------------------
        env = self

        # ----------------------------------------------------------------
        # 1. get_invoices — list invoice summaries
        # ----------------------------------------------------------------
        @mcp.tool()
        def get_invoices() -> str:
            """List all invoice summaries for the current episode."""
            if not env._episode_data:
                return json.dumps({"error": "No episode loaded. Call reset() first."})

            summaries = []
            for inv in env._episode_data["invoices"]:
                summaries.append({
                    "invoice_id": inv["invoice_id"],
                    "vendor_name": inv["vendor_name"],
                    "total_value": inv["total_value"],
                    "invoice_date": inv["invoice_date"],
                })

            # Also include fraud invoices for hard tasks
            if env._task_id == "hard":
                fraud_invs = env._episode_data.get("ground_truth", {}).get("fraud_invoices", [])
                for inv in fraud_invs:
                    summaries.append({
                        "invoice_id": inv["invoice_id"],
                        "vendor_name": inv["vendor_name"],
                        "total_value": inv["total_value"],
                        "invoice_date": inv["invoice_date"],
                    })

            env._step_reward = 0.01
            return json.dumps({"invoices": summaries, "count": len(summaries)})

        # ----------------------------------------------------------------
        # 2. get_invoice_details — full invoice data
        # ----------------------------------------------------------------
        @mcp.tool()
        def get_invoice_details(invoice_id: str) -> str:
            """Get full details for a specific invoice."""
            if not env._episode_data:
                return json.dumps({"error": "No episode loaded. Call reset() first."})

            # Search in regular invoices
            for inv in env._episode_data["invoices"]:
                if inv["invoice_id"] == invoice_id:
                    env._invoices_viewed.add(invoice_id)
                    env._step_reward = 0.01
                    return json.dumps(inv)

            # Search in fraud invoices (hard mode)
            if env._task_id == "hard":
                fraud_invs = env._episode_data.get("ground_truth", {}).get("fraud_invoices", [])
                for inv in fraud_invs:
                    if inv["invoice_id"] == invoice_id:
                        env._invoices_viewed.add(invoice_id)
                        env._step_reward = 0.01
                        return json.dumps(inv)

            env._step_reward = 0.0
            return json.dumps({"error": f"Invoice '{invoice_id}' not found."})

        # ----------------------------------------------------------------
        # 3. validate_gstin_tool — validate a GSTIN
        # ----------------------------------------------------------------
        @mcp.tool()
        def validate_gstin_tool(gstin: str) -> str:
            """Validate a GSTIN string and return validation result."""
            result = validate_gstin(gstin)
            if result["valid"]:
                env._step_reward = 0.02
            else:
                env._step_reward = 0.03  # finding invalid GSTIN is more valuable
            return json.dumps(result)

        # ----------------------------------------------------------------
        # 4. classify_hsn — lookup HSN code for a product
        # ----------------------------------------------------------------
        @mcp.tool()
        def classify_hsn(product_description: str) -> str:
            """Classify a product description to an HSN code."""
            result = lookup_hsn(product_description)
            if result is not None:
                env._step_reward = 0.05
                return json.dumps(result)
            else:
                env._step_reward = -0.02
                return json.dumps({"error": "No matching HSN code found.", "product_description": product_description})

        # ----------------------------------------------------------------
        # 5. compute_tax_tool — compute tax for an invoice
        # ----------------------------------------------------------------
        @mcp.tool()
        def compute_tax_tool(invoice_id: str, hsn_code: str, place_of_supply: str) -> str:
            """Compute GST tax for an invoice given HSN code and place of supply."""
            if not env._episode_data:
                return json.dumps({"error": "No episode loaded. Call reset() first."})

            # Find the invoice
            invoice = None
            for inv in env._episode_data["invoices"]:
                if inv["invoice_id"] == invoice_id:
                    invoice = inv
                    break

            # Also check fraud invoices
            if invoice is None and env._task_id == "hard":
                fraud_invs = env._episode_data.get("ground_truth", {}).get("fraud_invoices", [])
                for inv in fraud_invs:
                    if inv["invoice_id"] == invoice_id:
                        invoice = inv
                        break

            if invoice is None:
                env._step_reward = -0.03
                return json.dumps({"error": f"Invoice '{invoice_id}' not found."})

            # Lookup HSN rate
            if hsn_code in HSN_DATABASE:
                tax_rate = HSN_DATABASE[hsn_code]["tax_rate"]
            else:
                env._step_reward = -0.03
                return json.dumps({"error": f"HSN code '{hsn_code}' not found in database."})

            business_state = env._episode_data["business"]["state_code"]
            tax_result = compute_tax(invoice["taxable_value"], tax_rate, business_state, place_of_supply)

            # Store result for grading
            env._agent_results[invoice_id] = {
                "hsn_code": hsn_code,
                "is_interstate": tax_result["is_interstate"],
                "total_tax": tax_result["total_tax"],
                "cgst": tax_result["cgst"],
                "sgst": tax_result["sgst"],
                "igst": tax_result["igst"],
            }

            env._step_reward = 0.05
            return json.dumps({
                "invoice_id": invoice_id,
                "hsn_code": hsn_code,
                "taxable_value": invoice["taxable_value"],
                **tax_result,
            })

        # ----------------------------------------------------------------
        # 6. reconcile_invoice — match against GSTR-2B data
        # ----------------------------------------------------------------
        @mcp.tool()
        def reconcile_invoice(invoice_id: str) -> str:
            """Reconcile an invoice against GSTR-2B records."""
            if not env._episode_data:
                return json.dumps({"error": "No episode loaded. Call reset() first."})

            ground_truth = env._episode_data.get("ground_truth", {})
            gstr2b_records = ground_truth.get("gstr2b_records", {})

            if not gstr2b_records:
                env._step_reward = -0.03
                return json.dumps({
                    "error": "No GSTR-2B records available for this task.",
                    "hint": "GSTR-2B reconciliation is only available in hard mode."
                })

            # Find the invoice
            invoice = None
            for inv in env._episode_data["invoices"]:
                if inv["invoice_id"] == invoice_id:
                    invoice = inv
                    break

            if invoice is None:
                env._step_reward = -0.03
                return json.dumps({"error": f"Invoice '{invoice_id}' not found."})

            # Get GSTR-2B record
            gstr2b_record = gstr2b_records.get(invoice_id)
            if gstr2b_record is None:
                # Missing from GSTR-2B
                result = {
                    "invoice_id": invoice_id,
                    "match_status": "missing_from_supplier",
                    "invoice_taxable_value": invoice["taxable_value"],
                    "gstr2b_taxable_value": 0.0,
                    "difference": invoice["taxable_value"],
                }
                env._agent_reconciliation[invoice_id] = {
                    "match_status": "mismatch",
                    "reason": "missing_from_supplier",
                }
                env._step_reward = 0.05
                return json.dumps(result)

            # Compare values
            status = gstr2b_record.get("status", "matched")
            difference = round(invoice["taxable_value"] - gstr2b_record["taxable_value"], 2)

            result = {
                "invoice_id": invoice_id,
                "match_status": status,
                "invoice_taxable_value": invoice["taxable_value"],
                "gstr2b_taxable_value": gstr2b_record["taxable_value"],
                "invoice_total_tax": invoice["total_tax"],
                "gstr2b_total_tax": gstr2b_record["total_tax"],
                "difference": difference,
            }

            env._agent_reconciliation[invoice_id] = {
                "match_status": "mismatch" if status != "matched" else "match",
                "reason": status,
            }

            env._step_reward = 0.05 if status != "matched" else 0.02
            return json.dumps(result)

        # ----------------------------------------------------------------
        # 7. flag_invoice — flag as error or fraud
        # ----------------------------------------------------------------
        @mcp.tool()
        def flag_invoice(invoice_id: str, reason: str) -> str:
            """Flag an invoice as containing an error or being fraudulent."""
            if not env._episode_data:
                return json.dumps({"error": "No episode loaded. Call reset() first."})

            ground_truth = env._episode_data.get("ground_truth", {})

            # Check if this is a genuine invalid invoice (easy mode)
            invalid_invoices = ground_truth.get("invalid_invoices", {})
            if invoice_id in invalid_invoices:
                env._agent_flags[invoice_id] = reason
                env._step_reward = 0.08
                return json.dumps({
                    "invoice_id": invoice_id,
                    "status": "flagged",
                    "reason": reason,
                    "result": "Invoice flagged successfully."
                })

            # Check if this is a fraud invoice (hard mode)
            fraud_invoices = ground_truth.get("fraud_invoices", [])
            fraud_ids = {inv["invoice_id"] for inv in fraud_invoices} if isinstance(fraud_invoices, list) else set()
            if invoice_id in fraud_ids:
                env._agent_fraud_flags[invoice_id] = reason
                env._step_reward = 0.10
                return json.dumps({
                    "invoice_id": invoice_id,
                    "status": "flagged_fraud",
                    "reason": reason,
                    "result": "Fraud flag recorded."
                })

            # Also track flags for medium-mode wrong-HSN invoices
            correct_hsn = ground_truth.get("correct_hsn", {})
            if invoice_id in correct_hsn:
                env._agent_flags[invoice_id] = reason
                env._step_reward = 0.08
                return json.dumps({
                    "invoice_id": invoice_id,
                    "status": "flagged",
                    "reason": reason,
                    "result": "Invoice flagged successfully."
                })

            # False positive
            env._agent_flags[invoice_id] = reason
            env._step_reward = -0.05
            return json.dumps({
                "invoice_id": invoice_id,
                "status": "flagged",
                "reason": reason,
                "result": "Invoice flagged (no known issue found)."
            })

        # ----------------------------------------------------------------
        # 8. submit_return — grade and end episode
        # ----------------------------------------------------------------
        @mcp.tool()
        def submit_return(return_data: str) -> str:
            """Submit the GST return data for grading. Ends the episode."""
            if not env._episode_data:
                return json.dumps({"error": "No episode loaded. Call reset() first."})

            ground_truth = env._episode_data.get("ground_truth", {})
            invoices = env._episode_data.get("invoices", [])
            total_invoices = len(invoices)

            # Parse return_data if it's a JSON string
            try:
                if isinstance(return_data, str):
                    parsed_data = json.loads(return_data)
                else:
                    parsed_data = return_data
            except (json.JSONDecodeError, TypeError):
                parsed_data = {}

            # Grade based on task difficulty
            if env._task_id == "easy":
                score = grade_easy(env._agent_flags, ground_truth, total_invoices)
            elif env._task_id == "medium":
                score = grade_medium(env._agent_results, ground_truth)
            elif env._task_id == "hard":
                # Build ITC mismatches from gstr2b_records
                gstr2b_records = ground_truth.get("gstr2b_records", {})
                itc_mismatches = {}
                for inv_id, rec in gstr2b_records.items():
                    if rec.get("status") != "matched" and not inv_id.endswith("_DUP"):
                        itc_mismatches[inv_id] = rec["status"]

                hard_gt = {
                    "itc_mismatches": itc_mismatches,
                    "fraud_invoices": {inv["invoice_id"] for inv in ground_truth.get("fraud_invoices", [])},
                    "fraud_pattern": ground_truth.get("fraud_pattern", ""),
                }
                score = grade_hard(
                    env._agent_reconciliation,
                    env._agent_fraud_flags,
                    hard_gt,
                    total_invoices,
                )
            else:
                score = 0.0

            score = round(max(0.0, min(1.0, score)), 4)

            env._episode_done = True
            env._step_reward = score

            return json.dumps({
                "status": "graded",
                "task_id": env._task_id,
                "score": score,
                "invoices_viewed": len(env._invoices_viewed),
                "flags_submitted": len(env._agent_flags),
                "fraud_flags_submitted": len(env._agent_fraud_flags),
                "tax_computations": len(env._agent_results),
                "reconciliations": len(env._agent_reconciliation),
            })

        # ----------------------------------------------------------------
        # Initialize MCPEnvironment with the FastMCP server
        # ----------------------------------------------------------------
        super().__init__(mcp)

        # ----------------------------------------------------------------
        # State variables
        # ----------------------------------------------------------------
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_data: Optional[dict] = None
        self._task_id: str = "easy"
        self._episode_done: bool = False
        self._step_reward: float = 0.0
        self._call_history: list = []
        self._agent_flags: dict = {}
        self._agent_results: dict = {}
        self._agent_reconciliation: dict = {}
        self._agent_fraud_flags: dict = {}
        self._invoices_viewed: set = set()

    def _reset_state(self):
        """Reset all episode-level tracking state."""
        self._episode_done = False
        self._step_reward = 0.0
        self._call_history = []
        self._agent_flags = {}
        self._agent_results = {}
        self._agent_reconciliation = {}
        self._agent_fraud_flags = {}
        self._invoices_viewed = set()

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        """Reset the environment and generate new episode data."""
        self._task_id = kwargs.get("task_id", "easy")
        ep_id = episode_id or str(uuid4())

        # Use seed or generate one
        if seed is None:
            seed = hash(ep_id) % (2**31)

        self._state = State(episode_id=ep_id, step_count=0)
        self._reset_state()

        # Generate episode data
        self._episode_data = generate_episode_data(seed, self._task_id)
        business = self._episode_data["business"]
        invoices = self._episode_data["invoices"]

        # Build observation with business info and instructions
        instructions = (
            f"You are auditing GST compliance for {business['company_name']} "
            f"(GSTIN: {business['gstin']}) in {business['state']} "
            f"for the tax period {business['tax_period']}. "
            f"There are {len(invoices)} invoices to review. "
            f"Task difficulty: {self._task_id}. "
        )

        if self._task_id == "easy":
            instructions += (
                "Your task is to validate all invoices — check GSTINs, "
                "verify place of supply, and flag any invalid invoices. "
                "When done, submit the return."
            )
        elif self._task_id == "medium":
            instructions += (
                "Your task is to verify HSN classifications and tax computations. "
                "Some invoices may have incorrect HSN codes. "
                "Reclassify products, recompute taxes, and flag discrepancies. "
                "When done, submit the return."
            )
        elif self._task_id == "hard":
            instructions += (
                "Your task is to reconcile invoices against GSTR-2B data, "
                "detect ITC mismatches, and identify potential fraud (circular trading). "
                "Flag suspicious invoices and submit the return."
            )

        return GSTObservation(
            done=False,
            reward=0.0,
            task_id=self._task_id,
            business={
                "company_name": business["company_name"],
                "gstin": business["gstin"],
                "state": business["state"],
                "state_code": business["state_code"],
                "tax_period": business["tax_period"],
            },
            invoice_count=len(invoices),
            instructions=instructions,
            step_count=0,
        )

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        """Handle non-MCP actions (fallback)."""
        return Observation(
            done=self._episode_done,
            reward=0.0,
            metadata={"error": "Unknown action type. Use MCP tool calls."},
        )

    def step(self, action, timeout_s=None, **kwargs) -> Observation:
        """Execute an action, tracking state and rewards."""
        # Increment step count
        self._state.step_count += 1

        # Check if episode already done
        if self._episode_done:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "Episode already finished. Call reset() to start a new episode."},
            )

        # Check max steps — auto-grade with penalty
        if self._state.step_count > MAX_STEPS:
            self._episode_done = True
            return Observation(
                done=True,
                reward=-0.1,
                metadata={
                    "error": f"Maximum steps ({MAX_STEPS}) exceeded. Episode terminated.",
                    "timeout_penalty": -0.1,
                },
            )

        # Reset step reward before tool execution
        self._step_reward = 0.0

        # Detect repeated calls (for CallToolAction)
        from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
        repeat_penalty = 0.0
        if isinstance(action, CallToolAction):
            call_key = (action.tool_name, json.dumps(action.arguments, sort_keys=True))
            if call_key in self._call_history:
                repeat_penalty = -0.01
            self._call_history.append(call_key)

        # Delegate to parent (handles ListToolsAction and CallToolAction)
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Apply reward and done state to the observation
        obs.reward = round(self._step_reward + repeat_penalty, 4)
        obs.done = self._episode_done
        if not obs.metadata:
            obs.metadata = {}
        obs.metadata["step_count"] = self._state.step_count
        obs.metadata["task_id"] = self._task_id

        return obs

    async def step_async(self, action, timeout_s=None, **kwargs) -> Observation:
        """Async version of step()."""
        # Increment step count
        self._state.step_count += 1

        # Check if episode already done
        if self._episode_done:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "Episode already finished. Call reset() to start a new episode."},
            )

        # Check max steps
        if self._state.step_count > MAX_STEPS:
            self._episode_done = True
            return Observation(
                done=True,
                reward=-0.1,
                metadata={
                    "error": f"Maximum steps ({MAX_STEPS}) exceeded. Episode terminated.",
                    "timeout_penalty": -0.1,
                },
            )

        # Reset step reward
        self._step_reward = 0.0

        # Detect repeated calls
        from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
        repeat_penalty = 0.0
        if isinstance(action, CallToolAction):
            call_key = (action.tool_name, json.dumps(action.arguments, sort_keys=True))
            if call_key in self._call_history:
                repeat_penalty = -0.01
            self._call_history.append(call_key)

        # Delegate to parent async step
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        # Apply reward and done state
        obs.reward = round(self._step_reward + repeat_penalty, 4)
        obs.done = self._episode_done
        if not obs.metadata:
            obs.metadata = {}
        obs.metadata["step_count"] = self._state.step_count
        obs.metadata["task_id"] = self._task_id

        return obs

    @property
    def state(self) -> State:
        return self._state
