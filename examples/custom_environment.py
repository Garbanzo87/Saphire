"""Build an environment from your own data export, evaluate an agent on it, and train the router.

No server needed. Replace the CSV with a real export, or use SQLConnector / OpenAPIConnector /
registry_from_mcp to point at live systems.
"""
import tempfile
from pathlib import Path

from saphire.connectors import DataEnvironment, FileDataConnector
from saphire.evaluation.runner import evaluate
from saphire.sdk import AgentConfig, MockLLM, TaskSpec, ToolAgent
from saphire.training.learners import train_router

tmp = Path(tempfile.mkdtemp())
(tmp / "tickets.csv").write_text("ticket_id,subject,status\nTKT-1,Login broken,open\nTKT-2,Refund request,open\n")
conn = FileDataConnector({"tickets": tmp / "tickets.csv"}, keys={"tickets": "ticket_id"}, writable=("tickets",))
tasks = [
    TaskSpec(env_name="helpdesk", instruction="Get tickets TKT-1, then update tickets TKT-1 to status closed", tags=["close"],
             expected={"tools": ["get_tickets", "update_tickets"], "state_checks": [{"path": "tables.tickets.0.status", "equals": "closed"}]}),
    TaskSpec(env_name="helpdesk", instruction="List tickets and tell me which ones are open", tags=["list"],
             expected={"tools": ["list_tickets"], "answer_contains": ["TKT-2"]}),
]
env = DataEnvironment("helpdesk", conn.registry(), conn.seed_data(), tasks=tasks)
agent = ToolAgent(AgentConfig(name="helpdesk-agent", model="mock", router_top_k=0), llm=MockLLM())
rollouts = []
res = evaluate(agent, env.generate_tasks(), envs={"helpdesk": env}, on_rollout=lambda ro, t, r: rollouts.append(ro))
print({k: round(v, 3) for k, v in res.metrics.items() if k in ("task_success", "tool_selection_f1", "step_efficiency")})
print(train_router(rollouts, {"helpdesk": env}, tmp / "router"))
