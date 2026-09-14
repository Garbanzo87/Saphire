"""Built-in `support_desk` environment.

Simulates a company's customer-support stack from first-party data (customers, orders,
shipments, tickets, knowledge base) with ~30 tools: 14 that matter and 16 realistic
distractors. Tasks are generated from templates with ground truth so rollouts can be
verified from the *final world state*, not just from the transcript.

Task families:
  lookup          – single tool, easy
  cancel_email    – 2 ordered steps
  refund_resolve  – 2 steps with numeric argument
  multiturn       – information revealed across simulated user turns (context preservation)
  kb_email        – retrieval + action
  find_list       – argument passed from tool output to the next call
  damage_workflow – 6-step long-horizon workflow (hard)
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..sdk.tools import ToolRegistry
from ..sdk.types import Reward, Rollout, TaskSpec
from .base import Environment, EnvState, register_environment

FIRST = ["Aisha", "Bilal", "Chen", "Dana", "Elias", "Fatima", "Gabriel", "Hana", "Ibrahim", "Julia", "Khalid", "Lena",
         "Mateo", "Noor", "Omar", "Priya", "Quinn", "Rania", "Sami", "Tara"]
LAST = ["Ali", "Brown", "Costa", "Dubois", "Eriksen", "Farah", "Garcia", "Haddad", "Ito", "Jensen"]
PRODUCTS = [("SKU-1001", "Wireless Headphones", 129.0), ("SKU-1002", "Standing Desk", 449.0), ("SKU-1003", "Coffee Grinder", 79.0),
            ("SKU-1004", "Running Shoes", 119.0), ("SKU-1005", "Monitor 27in", 299.0), ("SKU-1006", "Backpack", 59.0)]
ADDRESSES = ["12 Harbor St, Boston MA", "88 Elm Ave, Austin TX", "5 Rue Lepic, Paris", "301 Bay Rd, Sydney NSW",
             "77 Kings Rd, London", "9 Marina Walk, Dubai"]
REASONS = ["it was a duplicate", "the customer changed their mind", "the item arrived damaged", "it shipped to the wrong address",
           "the delivery is too late"]
KB = [
    {"id": "KB-1", "title": "Return policy", "body": "Items can be returned within 30 days of delivery for a full refund."},
    {"id": "KB-2", "title": "Shipping times", "body": "Standard shipping takes 3-5 business days; express takes 1-2."},
    {"id": "KB-3", "title": "Warranty", "body": "All electronics carry a 12-month manufacturer warranty."},
    {"id": "KB-4", "title": "Password reset", "body": "Use the 'Forgot password' link on the login page to reset your password."},
    {"id": "KB-5", "title": "Loyalty program", "body": "Gold tier customers earn 2x points and free express shipping."},
]


def build_seed_data(n_customers: int = 40, seed: int = 7) -> dict[str, Any]:
    rng = random.Random(seed)
    customers, orders, shipments, tickets = {}, {}, {}, {}
    for i in range(n_customers):
        cid = f"CUS-{1000 + i}"
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        email = f"{name.lower().replace(' ', '.')}{i}@example.com"
        customers[cid] = {"customer_id": cid, "name": name, "email": email, "tier": rng.choice(["standard", "gold", "standard"]),
                          "address": rng.choice(ADDRESSES), "orders": []}
        for _ in range(rng.randint(1, 3)):
            oid = f"ORD-{2000 + len(orders)}"
            sku, pname, price = rng.choice(PRODUCTS)
            qty = rng.randint(1, 2)
            status = rng.choice(["processing", "shipped", "delivered", "processing"])
            orders[oid] = {"order_id": oid, "customer_id": cid, "items": [{"product_id": sku, "name": pname, "qty": qty, "price": price}],
                           "total": round(price * qty, 2), "status": status, "shipping_address": customers[cid]["address"],
                           "refunds": [], "discounts": [], "shipment_id": None}
            customers[cid]["orders"].append(oid)
            if status in ("shipped", "delivered"):
                sid = f"SHP-{4000 + len(shipments)}"
                shipments[sid] = {"shipment_id": sid, "order_id": oid, "carrier": rng.choice(["UPS", "DHL", "FedEx"]),
                                  "status": "in_transit" if status == "shipped" else "delivered", "eta_days": rng.randint(1, 5)}
                orders[oid]["shipment_id"] = sid
        if rng.random() < 0.5:
            tid = f"TKT-{3000 + len(tickets)}"
            tickets[tid] = {"ticket_id": tid, "customer_id": cid, "subject": rng.choice(["Late delivery", "Damaged item", "Billing question"]),
                            "status": "open", "priority": "medium", "escalated": False, "notes": []}
    return {"customers": customers, "orders": orders, "shipments": shipments, "tickets": tickets, "kb": KB,
            "emails": [], "sms": [], "store_credits": [], "callbacks": [], "reports": [], "tasks": [], "audit": []}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def _order(state: EnvState, order_id: str) -> dict:
        o = state.data["orders"].get(order_id)
        if o is None:
            raise KeyError(f"order {order_id} not found")
        return o

    # ---- core tools ----
    @reg.register(tags=["core"])
    def lookup_customer(state, customer_id: str) -> dict:
        """Look up a customer record (name, email, tier, address, orders) by customer id.

        Args:
            customer_id: The customer id, e.g. CUS-1001
        """
        c = state.data["customers"].get(customer_id)
        if c is None:
            raise KeyError(f"customer {customer_id} not found")
        state.record("lookup_customer", customer_id=customer_id)
        return {k: v for k, v in c.items()}

    @reg.register(tags=["core"])
    def find_customer_by_email(state, email: str) -> dict:
        """Find a customer record using their email address.

        Args:
            email: Email address of the customer
        """
        for c in state.data["customers"].values():
            if c["email"].lower() == email.lower():
                state.record("find_customer_by_email", email=email)
                return c
        raise KeyError(f"no customer with email {email}")

    @reg.register(tags=["core"])
    def lookup_order(state, order_id: str) -> dict:
        """Look up an order: items, total, status, shipping address, refunds.

        Args:
            order_id: The order id, e.g. ORD-2001
        """
        state.record("lookup_order", order_id=order_id)
        return _order(state, order_id)

    @reg.register(tags=["core"])
    def list_customer_orders(state, customer_id: str) -> list:
        """List all orders placed by a customer.

        Args:
            customer_id: The customer id
        """
        c = state.data["customers"].get(customer_id)
        if c is None:
            raise KeyError(f"customer {customer_id} not found")
        state.record("list_customer_orders", customer_id=customer_id)
        return [state.data["orders"][o] for o in c["orders"]]

    @reg.register(tags=["core"])
    def issue_refund(state, order_id: str, amount: float, reason: str) -> dict:
        """Issue a refund to the customer's original payment method for an order.

        Args:
            order_id: The order to refund
            amount: Refund amount in dollars
            reason: Why the refund is issued
        """
        o = _order(state, order_id)
        if amount <= 0 or amount > o["total"] + 1e-6:
            raise ValueError(f"invalid refund amount {amount} for order total {o['total']}")
        o["refunds"].append({"amount": float(amount), "reason": reason})
        state.record("issue_refund", order_id=order_id, amount=float(amount), reason=reason)
        return {"order_id": order_id, "refunded": float(amount), "status": "refund_issued"}

    @reg.register(tags=["core"])
    def cancel_order(state, order_id: str, reason: str) -> dict:
        """Cancel an order that has not been delivered yet.

        Args:
            order_id: The order to cancel
            reason: Reason for the cancellation
        """
        o = _order(state, order_id)
        if o["status"] == "delivered":
            raise ValueError("delivered orders cannot be cancelled; issue a refund instead")
        o["status"] = "cancelled"
        state.record("cancel_order", order_id=order_id, reason=reason)
        return {"order_id": order_id, "status": "cancelled"}

    @reg.register(tags=["core"])
    def update_shipping_address(state, order_id: str, address: str) -> dict:
        """Change the shipping address of an order that has not shipped yet.

        Args:
            order_id: The order to update
            address: The new shipping address
        """
        o = _order(state, order_id)
        o["shipping_address"] = address
        state.record("update_shipping_address", order_id=order_id, address=address)
        return {"order_id": order_id, "shipping_address": address}

    @reg.register(tags=["core"])
    def track_shipment(state, order_id: str) -> dict:
        """Get delivery estimate, carrier and tracking status of the shipment for an order.

        Args:
            order_id: The order whose shipment to track
        """
        o = _order(state, order_id)
        state.record("track_shipment", order_id=order_id)
        if not o["shipment_id"]:
            return {"order_id": order_id, "status": o["status"], "eta_days": None, "note": "not shipped yet"}
        return state.data["shipments"][o["shipment_id"]]

    @reg.register(tags=["core"])
    def create_ticket(state, customer_id: str, subject: str, priority: str = "medium") -> dict:
        """Create a support ticket for a customer.

        Args:
            customer_id: The customer the ticket is for
            subject: Short description of the issue
            priority: low, medium, high or urgent
        """
        if customer_id not in state.data["customers"]:
            raise KeyError(f"customer {customer_id} not found")
        tid = f"TKT-{3000 + len(state.data['tickets'])}"
        t = {"ticket_id": tid, "customer_id": customer_id, "subject": subject, "status": "open", "priority": priority,
             "escalated": False, "notes": []}
        state.data["tickets"][tid] = t
        state.record("create_ticket", customer_id=customer_id, subject=subject, priority=priority, ticket_id=tid)
        return t

    @reg.register(tags=["core"])
    def update_ticket_status(state, ticket_id: str, status: str) -> dict:
        """Update the status of a support ticket (open, pending, resolved, closed).

        Args:
            ticket_id: The ticket id, e.g. TKT-3001
            status: New status
        """
        t = state.data["tickets"].get(ticket_id)
        if t is None:
            raise KeyError(f"ticket {ticket_id} not found")
        t["status"] = status
        state.record("update_ticket_status", ticket_id=ticket_id, status=status)
        return t

    @reg.register(tags=["core"])
    def escalate_ticket(state, ticket_id: str, reason: str) -> dict:
        """Escalate a support ticket to a senior agent.

        Args:
            ticket_id: The ticket to escalate
            reason: Why it needs escalation
        """
        t = state.data["tickets"].get(ticket_id)
        if t is None:
            raise KeyError(f"ticket {ticket_id} not found")
        t["escalated"] = True
        t["priority"] = "high"
        state.record("escalate_ticket", ticket_id=ticket_id, reason=reason)
        return t

    @reg.register(tags=["core"])
    def send_email(state, email: str, subject: str, body: str) -> dict:
        """Send an email to a customer.

        Args:
            email: Recipient email address
            subject: Email subject
            body: Email body
        """
        state.data["emails"].append({"to": email, "subject": subject, "body": body})
        state.record("send_email", email=email, subject=subject)
        return {"sent": True, "to": email}

    @reg.register(tags=["core"])
    def search_knowledge_base(state, query: str) -> list:
        """Search the help-center knowledge base articles for a policy or how-to question.

        Args:
            query: Free-text search query
        """
        q = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for a in state.data["kb"]:
            toks = set(re.findall(r"[a-z]+", (a["title"] + " " + a["body"]).lower()))
            scored.append((len(q & toks), a))
        scored.sort(key=lambda x: -x[0])
        state.record("search_knowledge_base", query=query)
        return [a for s, a in scored[:2] if s > 0]

    @reg.register(tags=["core"])
    def apply_discount_code(state, order_id: str, code: str) -> dict:
        """Apply a promotional discount code to an order.

        Args:
            order_id: The order
            code: The discount code
        """
        o = _order(state, order_id)
        o["discounts"].append(code)
        state.record("apply_discount_code", order_id=order_id, code=code)
        return {"order_id": order_id, "discounts": o["discounts"]}

    # ---- distractors (realistic, similarly named) ----
    def _side(name: str, doc: str):
        def _f(state, **kw):
            state.record(name, **kw)
            return {"ok": True, "tool": name, **kw}

        _f.__name__ = name
        _f.__doc__ = doc
        return _f

    distractors = {
        "lookup_vendor": ("vendor_id: Vendor id", "Look up a supplier/vendor record by vendor id.\n\nArgs:\n    vendor_id: Vendor id"),
        "lookup_invoice": ("invoice_id", "Look up an accounting invoice by invoice id.\n\nArgs:\n    invoice_id: Invoice id, e.g. INV-1"),
        "issue_store_credit": ("customer_id, amount", "Issue store credit to a customer's wallet instead of a refund.\n\nArgs:\n    customer_id: Customer id\n    amount: Credit amount"),
        "refund_invoice": ("invoice_id, amount", "Refund a B2B invoice payment.\n\nArgs:\n    invoice_id: Invoice id\n    amount: Amount"),
        "cancel_subscription": ("customer_id, reason", "Cancel a customer's recurring subscription plan.\n\nArgs:\n    customer_id: Customer id\n    reason: Reason"),
        "update_billing_address": ("customer_id, address", "Update the billing address on a customer's payment profile.\n\nArgs:\n    customer_id: Customer id\n    address: New billing address"),
        "track_return": ("return_id", "Track the status of a product return shipment.\n\nArgs:\n    return_id: Return id"),
        "create_task": ("subject, assignee", "Create an internal task for the operations team.\n\nArgs:\n    subject: Task subject\n    assignee: Team member"),
        "escalate_to_legal": ("ticket_id, reason", "Escalate a case to the legal department.\n\nArgs:\n    ticket_id: Ticket id\n    reason: Reason"),
        "send_sms": ("phone, message", "Send an SMS text message to a phone number.\n\nArgs:\n    phone: Phone number\n    message: Text"),
        "search_orders_archive": ("query", "Search archived (older than 2 years) orders.\n\nArgs:\n    query: Search query"),
        "list_warehouse_stock": ("product_id", "List warehouse stock levels for a product.\n\nArgs:\n    product_id: SKU"),
        "close_account": ("customer_id, reason", "Permanently close a customer account.\n\nArgs:\n    customer_id: Customer id\n    reason: Reason"),
        "reopen_ticket": ("ticket_id", "Reopen a previously closed support ticket.\n\nArgs:\n    ticket_id: Ticket id"),
        "schedule_callback": ("customer_id, time", "Schedule a phone callback with a customer.\n\nArgs:\n    customer_id: Customer id\n    time: Time slot"),
        "generate_report": ("report_type", "Generate a management report (sales, tickets, refunds).\n\nArgs:\n    report_type: Report type"),
    }
    for name, (params, doc) in distractors.items():
        fn = _side(name, doc)
        props = {}
        required = []
        for p in [p.strip().split(":")[0] for p in params.split(",")]:
            props[p] = {"type": "number" if p == "amount" else "string"}
            required.append(p)
        from ..sdk.types import ToolSpec

        spec = ToolSpec(name=name, description=doc.split("\n")[0], parameters={"type": "object", "properties": props, "required": required},
                        tags=["distractor"])
        reg.add_spec(spec, fn)
        reg.get(name).takes_state = True
    return reg


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@register_environment
class SupportDeskEnv(Environment):
    name = "support_desk"
    description = "Customer support desk over first-party CRM/order data with 30 tools (14 core + 16 distractors)."

    def __init__(self, n_customers: int = 40, data_seed: int = 7):
        super().__init__()
        self._seed = build_seed_data(n_customers, data_seed)

    def build_tools(self) -> ToolRegistry:
        return build_registry()

    def seed_data(self) -> dict[str, Any]:
        return self._seed

    # ---------------- task generation ----------------
    def generate_tasks(self, n: int = 20, seed: int = 0, families: list[str] | None = None) -> list[TaskSpec]:
        rng = random.Random(seed)
        d = self._seed
        fams = families or ["lookup", "cancel_email", "refund_resolve", "multiturn", "kb_email", "find_list", "damage_workflow"]
        tasks: list[TaskSpec] = []
        orders = list(d["orders"].values())
        tickets = list(d["tickets"].values())
        for i in range(n):
            fam = fams[i % len(fams)]
            o = rng.choice(orders)
            c = d["customers"][o["customer_id"]]
            if fam == "lookup":
                tasks.append(TaskSpec(env_name=self.name, difficulty="easy", tags=[fam],
                                      instruction=f"Look up order {o['order_id']} and report its current status.",
                                      expected={"tools": ["lookup_order"], "answer_contains": [o["status"]]}, max_steps=4))
            elif fam == "cancel_email":
                o2 = rng.choice([x for x in orders if x["status"] != "delivered"])
                c2 = d["customers"][o2["customer_id"]]
                reason = rng.choice(REASONS)
                tasks.append(TaskSpec(env_name=self.name, difficulty="medium", tags=[fam],
                                      instruction=f"Cancel order {o2['order_id']} because {reason}, then send an email to {c2['email']} confirming the cancellation.",
                                      expected={"tools": ["cancel_order", "send_email"], "order_status": {o2["order_id"]: "cancelled"},
                                                "email_to": c2["email"]}, max_steps=6))
            elif fam == "refund_resolve":
                t = rng.choice(tickets)
                amt = round(min(o["total"], rng.choice([10, 15, 20, 25, 40])), 2)
                tasks.append(TaskSpec(env_name=self.name, difficulty="medium", tags=[fam],
                                      instruction=f"Issue a refund of ${amt:.2f} on order {o['order_id']} because {rng.choice(REASONS)}, then update ticket {t['ticket_id']} to resolved.",
                                      expected={"tools": ["issue_refund", "update_ticket_status"], "refund": {o["order_id"]: amt},
                                                "ticket_status": {t["ticket_id"]: "resolved"}}, max_steps=6))
            elif fam == "multiturn":
                o3 = rng.choice([x for x in orders if x["status"] == "shipped"])
                addr = rng.choice(ADDRESSES)
                tasks.append(TaskSpec(env_name=self.name, difficulty="hard", tags=[fam, "context"],
                                      instruction=f"Hi, I need help with my order {o3['order_id']}.",
                                      user_script=[f"Please change the shipping address to {addr}",
                                                   "Also, please track the shipment for me and tell me when it will arrive."],
                                      expected={"tools": ["update_shipping_address", "track_shipment"],
                                                "address": {o3["order_id"]: addr}, "tracked": [o3["order_id"]],
                                                "context_values": {"order_id": o3["order_id"]}}, max_steps=8))
            elif fam == "kb_email":
                topic, kb_id = rng.choice([("return policy", "KB-1"), ("shipping times", "KB-2"), ("warranty", "KB-3"), ("loyalty program", "KB-5")])
                tasks.append(TaskSpec(env_name=self.name, difficulty="medium", tags=[fam],
                                      instruction=f"Search the knowledge base for our {topic}, then send an email to {c['email']} with the answer.",
                                      expected={"tools": ["search_knowledge_base", "send_email"], "email_to": c["email"], "kb": kb_id}, max_steps=6))
            elif fam == "find_list":
                tasks.append(TaskSpec(env_name=self.name, difficulty="medium", tags=[fam, "context"],
                                      instruction=f"Find the customer with email {c['email']}, then list all orders for that customer.",
                                      expected={"tools": ["find_customer_by_email", "list_customer_orders"],
                                                "listed": [c["customer_id"]], "context_values": {"customer_id": c["customer_id"]}}, max_steps=6))
            elif fam == "damage_workflow":
                amt = round(min(o["total"], 30.0), 2)
                tasks.append(TaskSpec(env_name=self.name, difficulty="hard", tags=[fam, "long_horizon"],
                                      instruction=(f"Customer {c['customer_id']} reports a damaged item in order {o['order_id']}. "
                                                   f"Look up the customer {c['customer_id']}, then look up the order {o['order_id']}, "
                                                   f"then create a high priority ticket for customer {c['customer_id']} about the damaged item, "
                                                   f"then issue a refund of ${amt:.2f} on order {o['order_id']} because the item arrived damaged, "
                                                   f"then escalate the ticket, then send an email to {c['email']} with an apology."),
                                      expected={"tools": ["lookup_customer", "lookup_order", "create_ticket", "issue_refund", "escalate_ticket", "send_email"],
                                                "ordered": True, "refund": {o["order_id"]: amt}, "ticket_for": c["customer_id"],
                                                "escalated_for": c["customer_id"], "email_to": c["email"],
                                                "context_values": {"ticket_id": "<created>"}}, max_steps=12))
        return tasks

    # ---------------- verification ----------------
    def verify(self, task: TaskSpec, rollout: Rollout, state: EnvState) -> list[Reward]:
        exp = task.expected
        d = state.data
        checks: dict[str, bool] = {}
        called = [tc.name for tc in rollout.tool_calls]
        # tool selection: set-level F1 + order check
        exp_tools = exp.get("tools", [])
        if exp_tools:
            es, cs = set(exp_tools), set(called)
            tp = len(es & cs)
            prec = tp / len(cs) if cs else 0.0
            rec = tp / len(es)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            checks["all_expected_tools_called"] = es <= cs
            if exp.get("ordered"):
                seq = [n for n in called if n in es]
                # expected order as subsequence
                it = iter(seq)
                checks["tool_order"] = all(any(x == e for x in it) for e in exp_tools)
        else:
            f1 = 1.0
        # state-based checks
        for oid, st in exp.get("order_status", {}).items():
            checks[f"order_{oid}_status"] = d["orders"][oid]["status"] == st
        for oid, amt in exp.get("refund", {}).items():
            checks[f"refund_{oid}"] = any(abs(r["amount"] - amt) < 0.01 for r in d["orders"][oid]["refunds"])
        for tid, st in exp.get("ticket_status", {}).items():
            checks[f"ticket_{tid}_status"] = d["tickets"][tid]["status"] == st
        for oid, addr in exp.get("address", {}).items():
            checks[f"address_{oid}"] = d["orders"][oid]["shipping_address"] == addr
        if "email_to" in exp:
            checks["email_sent"] = any(e["to"].lower() == exp["email_to"].lower() for e in d["emails"])
        for oid in exp.get("tracked", []):
            checks[f"tracked_{oid}"] = any(entry["action"] == "track_shipment" and entry.get("order_id") == oid for entry in state.log)
        for cid in exp.get("listed", []):
            checks[f"listed_{cid}"] = any(entry["action"] == "list_customer_orders" and entry.get("customer_id") == cid for entry in state.log)
        if "ticket_for" in exp:
            created = [entry for entry in state.log if entry["action"] == "create_ticket" and entry.get("customer_id") == exp["ticket_for"]]
            checks["ticket_created"] = bool(created)
            if "escalated_for" in exp:
                checks["ticket_escalated"] = any(d["tickets"][entry["ticket_id"]]["escalated"] for entry in created)
        if "kb" in exp:
            checks["kb_searched"] = any(entry["action"] == "search_knowledge_base" for entry in state.log)
        for s in exp.get("answer_contains", []):
            checks[f"answer_contains_{s}"] = s.lower() in rollout.final_answer.lower()
        # side-effect safety: distractor / destructive tools not expected
        harmful = [entry["action"] for entry in state.log if entry["action"] in ("close_account", "cancel_subscription", "escalate_to_legal", "issue_store_credit")]
        checks["no_harmful_side_effects"] = not harmful
        # errors from tools
        n_err = sum(1 for s in rollout.steps for r in s.tool_results if r.error)

        success = all(checks.values()) and rollout.status.value in ("succeeded",)
        rewards = [
            Reward(value=1.0 if success else 0.0, source="verifier", name="task_success",
                   rationale="; ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()), metadata={"checks": checks}),
            Reward(value=f1, source="verifier", name="tool_selection_f1", metadata={"expected": exp_tools, "called": called}),
        ]
        if exp_tools:
            # step efficiency: expected tool calls / actual tool calls (capped at 1)
            rewards.append(Reward(value=min(1.0, len(exp_tools) / max(1, len(called))), source="verifier", name="step_efficiency"))
        if "context_values" in exp:
            # were earlier-turn / earlier-tool values reused correctly?
            ok = True
            for k, v in exp["context_values"].items():
                if v == "<created>":
                    created_ids = {entry["ticket_id"] for entry in state.log if entry["action"] == "create_ticket"}
                    used = {tc.arguments.get("ticket_id") for tc in rollout.tool_calls if tc.name == "escalate_ticket"}
                    ok &= bool(created_ids & used)
                else:
                    later = [tc for tc in rollout.tool_calls if k in tc.arguments and tc.name in exp_tools]
                    ok &= bool(later) and all(tc.arguments.get(k) == v for tc in later)
            rewards.append(Reward(value=1.0 if ok else 0.0, source="verifier", name="context_preservation"))
        rewards.append(Reward(value=1.0 if n_err == 0 else max(0.0, 1 - 0.5 * n_err), source="verifier", name="tool_error_free"))
        # per-step tool-correctness signal (for routers / step-level RL)
        for s in rollout.steps:
            for tc in s.response.tool_calls:
                rewards.append(Reward(value=1.0 if tc.name in exp_tools else -1.0, source="verifier", name="tool_correct",
                                      step_index=s.index, metadata={"tool": tc.name}))
        return rewards
