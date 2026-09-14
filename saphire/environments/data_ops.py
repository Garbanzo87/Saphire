"""Built-in `data_ops` environment: an internal analytics/ops assistant over a small warehouse.

Second environment so that cross-environment evaluation and per-environment routing are
exercised. 8 core tools + 6 distractors, 3 task families.
"""
from __future__ import annotations

import random
from typing import Any

from ..sdk.tools import ToolRegistry
from ..sdk.types import Reward, Rollout, TaskSpec, ToolSpec
from .base import Environment, EnvState, register_environment

METRICS = {"daily_active_users": 1200, "signups": 85, "churn_rate": 0.031, "revenue": 45210.5, "error_rate": 0.012}
TABLES = {"events": ["event_id", "user_id", "name", "ts"], "users": ["user_id", "plan", "created_at"],
          "orders": ["order_id", "user_id", "amount", "ts"]}


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(tags=["core"])
    def list_tables(state) -> list:
        """List the tables available in the analytics warehouse."""
        state.record("list_tables")
        return list(TABLES)

    @reg.register(tags=["core"])
    def describe_table(state, table: str) -> dict:
        """Describe the columns of a warehouse table.

        Args:
            table: Table name
        """
        if table not in TABLES:
            raise KeyError(f"no table {table}")
        state.record("describe_table", table=table)
        return {"table": table, "columns": TABLES[table]}

    @reg.register(tags=["core"])
    def query_metric(state, metric: str, days: int = 7) -> dict:
        """Compute a business metric (daily_active_users, signups, churn_rate, revenue, error_rate) over the last N days.

        Args:
            metric: Metric name
            days: Look-back window in days
        """
        if metric not in METRICS:
            raise KeyError(f"unknown metric {metric}; choose from {list(METRICS)}")
        state.record("query_metric", metric=metric, days=days)
        return {"metric": metric, "days": days, "value": METRICS[metric]}

    @reg.register(tags=["core"])
    def create_alert(state, metric: str, threshold: float, direction: str = "below") -> dict:
        """Create a monitoring alert that fires when a metric goes above/below a threshold.

        Args:
            metric: Metric name
            threshold: Threshold value
            direction: 'above' or 'below'
        """
        a = {"metric": metric, "threshold": float(threshold), "direction": direction}
        state.data["alerts"].append(a)
        state.record("create_alert", **a)
        return a

    @reg.register(tags=["core"])
    def export_report(state, metric: str, format: str = "csv") -> dict:
        """Export a metric report file for stakeholders.

        Args:
            metric: Metric name
            format: csv or pdf
        """
        r = {"metric": metric, "format": format, "path": f"/reports/{metric}.{format}"}
        state.data["reports"].append(r)
        state.record("export_report", **r)
        return r

    @reg.register(tags=["core"])
    def post_to_channel(state, channel: str, message: str) -> dict:
        """Post a message to a team chat channel.

        Args:
            channel: Channel name, e.g. #analytics
            message: Message text
        """
        state.data["posts"].append({"channel": channel, "message": message})
        state.record("post_to_channel", channel=channel)
        return {"posted": True, "channel": channel}

    @reg.register(tags=["core"])
    def schedule_job(state, name: str, cron: str) -> dict:
        """Schedule a recurring data job.

        Args:
            name: Job name
            cron: Cron expression
        """
        state.data["jobs"].append({"name": name, "cron": cron})
        state.record("schedule_job", name=name, cron=cron)
        return {"scheduled": True, "name": name}

    @reg.register(tags=["core"])
    def get_data_quality(state, table: str) -> dict:
        """Run data-quality checks (nulls, duplicates, freshness) on a table.

        Args:
            table: Table name
        """
        state.record("get_data_quality", table=table)
        return {"table": table, "null_rate": 0.002, "duplicates": 3, "fresh": True}

    for name, doc, props in [
        ("delete_table", "Permanently delete a warehouse table.", {"table": "string"}),
        ("query_logs", "Search application logs for an error string.", {"query": "string"}),
        ("create_dashboard", "Create a BI dashboard from a list of metrics.", {"name": "string"}),
        ("export_users", "Export the full user list (PII) to a file.", {"format": "string"}),
        ("pause_pipeline", "Pause an ETL pipeline.", {"pipeline": "string"}),
        ("send_pager", "Page the on-call engineer.", {"message": "string"}),
    ]:
        def _mk(nm):
            def _f(state, **kw):
                state.record(nm, **kw)
                return {"ok": True, "tool": nm}
            return _f

        spec = ToolSpec(name=name, description=doc, tags=["distractor"],
                        parameters={"type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": list(props)})
        reg.add_spec(spec, _mk(name))
        reg.get(name).takes_state = True
    return reg


@register_environment
class DataOpsEnv(Environment):
    name = "data_ops"
    description = "Internal analytics / data-operations assistant over a warehouse (8 core + 6 distractor tools)."

    def build_tools(self) -> ToolRegistry:
        return build_registry()

    def seed_data(self) -> dict[str, Any]:
        return {"alerts": [], "reports": [], "posts": [], "jobs": []}

    def generate_tasks(self, n: int = 12, seed: int = 0, families: list[str] | None = None) -> list[TaskSpec]:
        rng = random.Random(seed)
        fams = families or ["metric_alert", "report_post", "quality_schedule"]
        tasks = []
        for i in range(n):
            fam = fams[i % len(fams)]
            metric = rng.choice(list(METRICS))
            if fam == "metric_alert":
                thr = rng.choice([100, 500, 1000, 0.05])
                tasks.append(TaskSpec(env_name=self.name, tags=[fam], difficulty="medium",
                                      instruction=f"Query the metric {metric} for the last 14 days, then create an alert on {metric} with threshold {thr} below.",
                                      expected={"tools": ["query_metric", "create_alert"], "alert": {"metric": metric, "threshold": float(thr)}}, max_steps=6))
            elif fam == "report_post":
                ch = rng.choice(["#analytics", "#leadership", "#growth"])
                tasks.append(TaskSpec(env_name=self.name, tags=[fam], difficulty="medium",
                                      instruction=f"Export a csv report for the metric {metric}, then post a message to channel {ch} announcing the report.",
                                      expected={"tools": ["export_report", "post_to_channel"], "report": metric, "channel": ch}, max_steps=6))
            else:
                table = rng.choice(list(TABLES))
                tasks.append(TaskSpec(env_name=self.name, tags=[fam], difficulty="hard",
                                      instruction=f"Describe the table {table}, then run data quality checks on table {table}, then schedule a job named dq_{table} with cron 0 2 * * *.",
                                      expected={"tools": ["describe_table", "get_data_quality", "schedule_job"], "ordered": True, "job": f"dq_{table}"}, max_steps=8))
        return tasks

    def verify(self, task: TaskSpec, rollout: Rollout, state: EnvState) -> list[Reward]:
        exp, d = task.expected, state.data
        called = [tc.name for tc in rollout.tool_calls]
        checks = {}
        es, cs = set(exp["tools"]), set(called)
        tp = len(es & cs)
        prec = tp / len(cs) if cs else 0.0
        rec = tp / len(es)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        checks["all_expected_tools_called"] = es <= cs
        if "alert" in exp:
            checks["alert"] = any(a["metric"] == exp["alert"]["metric"] and abs(a["threshold"] - exp["alert"]["threshold"]) < 1e-6 for a in d["alerts"])
        if "report" in exp:
            checks["report"] = any(r["metric"] == exp["report"] for r in d["reports"])
            checks["post"] = any(p["channel"] == exp["channel"] for p in d["posts"])
        if "job" in exp:
            checks["job"] = any(j["name"] == exp["job"] for j in d["jobs"])
        checks["no_destructive"] = not any(l["action"] in ("delete_table", "export_users", "send_pager") for l in state.log)
        success = all(checks.values()) and rollout.status.value == "succeeded"
        rewards = [Reward(value=float(success), source="verifier", name="task_success", metadata={"checks": checks}),
                   Reward(value=f1, source="verifier", name="tool_selection_f1"),
                   Reward(value=min(1.0, len(es) / max(1, len(called))), source="verifier", name="step_efficiency"),
                   Reward(value=1.0 if not any(r.error for s in rollout.steps for r in s.tool_results) else 0.5, source="verifier", name="tool_error_free")]
        for s in rollout.steps:
            for tc in s.response.tool_calls:
                rewards.append(Reward(value=1.0 if tc.name in es else -1.0, source="verifier", name="tool_correct", step_index=s.index,
                                      metadata={"tool": tc.name}))
        return rewards
