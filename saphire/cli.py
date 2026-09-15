"""`saphire` command line: serve the API, run workers, evaluate, train, and run the demo."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Saphire – continuously learning agent stack.", no_args_is_help=True)
console = Console()


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False, inline_jobs: bool = typer.Option(False, help="run jobs inside the API process")):
    """Start the API server."""
    import uvicorn

    if inline_jobs:
        os.environ["SAPHIRE_INLINE_JOBS"] = "1"
    uvicorn.run("saphire.server.main:app", host=host, port=port, reload=reload)


@app.command("env-server")
def env_server(host: str = "0.0.0.0", port: int = 8010):
    """Standalone environment / reward server for remote RL trainers (verl, OpenRLHF, SkyRL)."""
    import uvicorn

    from .distributed.env_server import create_app

    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def worker(once: bool = False, poll: float = 1.0):
    """Start a background job worker (evaluations, training)."""
    from .server.jobs import worker_loop

    rprint("[green]worker started[/green]")
    n = worker_loop(poll_s=poll, once=once)
    if once:
        rprint(f"processed {n} job(s)")


@app.command("eval")
def eval_cmd(suite: str = "full", model: str = "mock", n_per_env: int = 14, k: int = 1, seed: int = 0, top_k: int = 8,
             router: Optional[str] = None, exemplars: Optional[str] = None, judge: Optional[str] = None, concurrency: int = 1,
             out: Optional[Path] = None, agent_json: Optional[Path] = None, multi_agent: bool = typer.Option(False, help="use the built-in orchestrator + specialists preset"),
             distributed: Optional[str] = typer.Option(None, help="rollout backend: ray | process | thread"), workers: int = 0):
    """Evaluate an agent locally (no server needed) on a named suite."""
    from .evaluation.runner import evaluate
    from .evaluation.suites import build_suite
    from .sdk.multi_agent import build_agent
    from .sdk.types import AgentConfig
    from .signals.judges import make_judge

    os.environ.setdefault("SAPHIRE_TRACING", "off")
    if agent_json:
        cfg = AgentConfig.model_validate_json(agent_json.read_text())
    elif multi_agent:
        from .environments.support_desk import multi_agent_config

        cfg = multi_agent_config(model=model, router_top_k=top_k)
    else:
        cfg = AgentConfig(model=model, router_top_k=top_k, tool_router=router, exemplar_store=exemplars)
    tasks = build_suite(suite, n_per_env=n_per_env, seed=seed)
    if distributed:
        from .distributed.rollouts import evaluate_distributed

        res = evaluate_distributed(cfg, tasks, k=k, backend=distributed, workers=workers or None, judge_model=judge, suite=suite)
    else:
        res = evaluate(build_agent(cfg), tasks, k=k, suite=suite, judge=make_judge(judge), concurrency=concurrency)
    if res.by_role:
        rprint({"by_role": res.by_role})
    _print_eval(res)
    if out:
        out.write_text(json.dumps(res.model_dump(), indent=2))
        rprint(f"saved {out}")


def _print_eval(res):
    m = res.metrics
    t = Table(title=f"{res.agent}/{res.agent_version} on {res.suite} ({res.n_rollouts} rollouts, k={res.k})")
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k in ["task_success", "pass_at_1", f"pass_hat_{res.k}", "tool_selection_f1", "context_preservation", "step_efficiency",
              "tool_error_free", "judge_score", "error_rate", "latency_ms_p50", "latency_ms_p95", "tokens_per_task", "throughput_tasks_per_min"]:
        if k in m:
            t.add_row(k, f"{m[k]:.3f}")
    console.print(t)
    t2 = Table(title="by family")
    t2.add_column("family")
    t2.add_column("n", justify="right")
    t2.add_column("success", justify="right")
    t2.add_column("tool f1", justify="right")
    for fam, mm in res.by_family.items():
        t2.add_row(fam, str(int(mm["n"])), f"{mm.get('task_success', 0):.2f}", f"{mm.get('tool_selection_f1', 0):.2f}")
    console.print(t2)


@app.command()
def train(algorithm: str = typer.Argument(..., help="online | prompt_opt | sft | dpo | grpo"), model: str = "mock:error=0.3",
          iterations: int = 5, batch_size: int = 24, n_train: int = 42, n_eval: int = 28, artifacts: Path = Path("artifacts"),
          learn_prompt_every: int = 2, seed: int = 1, base_model: str = "sshleifer/tiny-gpt2", max_steps: int = 5):
    """Run a training loop locally (no server needed)."""
    from .evaluation.suites import build_suite
    from .sdk.types import AgentConfig

    os.environ.setdefault("SAPHIRE_TRACING", "off")
    cfg = AgentConfig(name="local", model=model, router_top_k=8)
    train_tasks = build_suite("full", n_per_env=n_train, seed=100 + seed)
    eval_tasks = build_suite("full", n_per_env=n_eval, seed=200 + seed)
    if algorithm == "online":
        from .training.online import OnlineLoop

        loop = OnlineLoop(cfg, train_tasks, eval_tasks, artifacts_dir=artifacts, batch_size=batch_size,
                          learn_prompt_every=learn_prompt_every, seed=seed, on_iteration=_print_iter)
        loop.run(iterations)
        rprint(f"[green]final agent config:[/green] {artifacts / cfg.name / loop.config.version / 'agent.json'}")
    elif algorithm == "prompt_opt":
        from .training.prompt_opt import optimize_prompt

        r = optimize_prompt(cfg, train_tasks, iterations=iterations, seed=seed)
        rprint({k: v for k, v in r.items() if k != "history"})
    elif algorithm in ("sft", "dpo", "grpo"):
        from .environments.base import get_environment
        from .evaluation.runner import evaluate
        from .sdk.agent import ToolAgent
        from .sdk.llm import MockLLM
        from .signals.generate import grpo_prompts, preference_pairs, sft_examples
        from .training import trl_trainers as T

        envs = {e: get_environment(e) for e in {t.env_name for t in train_tasks}}
        tools_by_env = {k: v.tools.specs() for k, v in envs.items()}
        rollouts = []
        for err in (0.0, 0.5):
            evaluate(ToolAgent(cfg, llm=MockLLM(error_rate=err, seed=seed)), train_tasks[:batch_size], envs=envs,
                     on_rollout=lambda ro, t, r: rollouts.append(ro))
        rows = {"sft": lambda: sft_examples(rollouts, tools_by_env), "dpo": lambda: preference_pairs(rollouts, tools_by_env),
                "grpo": lambda: grpo_prompts(rollouts, {t.id: t for t in train_tasks}, tools_by_env)}[algorithm]()
        rprint(f"{len(rows)} {algorithm} rows generated from {len(rollouts)} rollouts")
        r = getattr(T, algorithm)(rows, artifacts / algorithm, model=base_model, max_steps=max_steps)
        rprint(r)
    else:
        raise typer.BadParameter("algorithm must be online | prompt_opt | sft | dpo | grpo")


def _print_iter(rec):
    e = rec["eval"]
    rprint(f"iter {rec['iteration']:>2} {rec['version']:<4} collect={rec['collect_success'] if rec['collect_success'] is None else round(rec['collect_success'], 2)} "
           f"success={e['task_success']:.3f} tool_f1={e['tool_selection_f1']:.3f} ctx={e.get('context_preservation', float('nan')):.2f} "
           f"p95={e['latency_ms_p95']:.0f}ms ({rec['seconds']:.1f}s)")


@app.command()
def demo(host: str = "http://localhost:8000", api_key: str = "dev-key", project: str = "default", iterations: int = 5,
         model: str = "mock:error=0.3,seed=3", wait: bool = True):
    """Populate a running server with the end-to-end demo: agent -> baseline eval -> online learning -> gated deployment."""
    from .sdk.client import SaphireClient
    from .sdk.types import AgentConfig

    c = SaphireClient(host, api_key, project)
    rprint(c.health())
    train_ds = c.create_dataset("support+dataops train", suite="full", n_per_env=42, seed=101, split="train")
    eval_ds = c.create_dataset("support+dataops heldout", suite="full", n_per_env=28, seed=202)
    ctx_ds = c.create_dataset("context preservation", suite="context_preservation", n_per_env=20, seed=303)
    agent = c.create_agent(AgentConfig(name="support-agent", version="v0", model=model, router_top_k=8, exemplar_k=2), status="deployed")
    rprint(f"agent {agent['id']} datasets {train_ds['id']} {eval_ds['id']}")
    ev = c.run_eval(agent["id"], eval_ds["id"], k=2, judge_model="mock", wait=wait)
    if wait:
        rprint("baseline:", {k: round(v, 3) for k, v in ev["metrics"].items() if k in ("task_success", "tool_selection_f1", "pass_hat_2")})
    tr = c.train(agent["id"], "online", train_dataset_id=train_ds["id"], eval_dataset_id=eval_ds["id"], iterations=iterations,
                 batch_size=24, learn_prompt_every=2, seed=1, wait=wait)
    if wait:
        rprint("improvement:", tr["result"]["improvement"])
        out_agent = tr["output_agent_id"]
        ev2 = c.run_eval(out_agent, eval_ds["id"], k=2, judge_model="mock", gate=True, auto_promote=True, wait=True)
        rprint("candidate:", {k: round(v, 3) for k, v in ev2["metrics"].items() if k in ("task_success", "tool_selection_f1", "pass_hat_2")})
        c.run_eval(out_agent, ctx_ds["id"], k=1, wait=True)
        deps = c.get("/deployments")
        rprint("gate:", deps[0]["decision"]["passed"], deps[0]["decision"]["reasons"])
        exp = c.post("/experiments", json={"name": "v0 vs learned", "variants": [{"name": "control", "agent_id": agent["id"], "weight": 1},
                                                                                 {"name": "learned", "agent_id": out_agent, "weight": 1}]})
        import random

        rng = random.Random(0)
        for i in range(120):
            a = c.assign_variant(exp["id"], f"user-{i}")
            p = 0.55 if a["variant"] == "control" else 0.8
            c.record_outcome(exp["id"], f"user-{i}", 1.0 if rng.random() < p else 0.0, a["variant"])
        rprint("experiment:", c.get(f"/experiments/{exp['id']}")["results"])
        # multi-agent system: eval -> per-role attribution (blame / advantage / ablation / Shapley)
        from .environments.support_desk import multi_agent_config

        ma_ds = c.create_dataset("support multi-agent", suite="tool_selection", n_per_env=16, seed=404)
        ma = c.create_agent(multi_agent_config(model="mock:error=0.35,seed=5", name="support-system"))
        c.run_eval(ma["id"], ma_ds["id"], k=1, judge_model="mock", wait=True)
        attr = c.post("/attribution", json={"agent_id": ma["id"], "dataset_id": ma_ds["id"], "reference_model": "mock",
                                            "degraded_model": "mock:error=0.9", "shapley": True, "n_permutations": 4})
        attr = c.wait_job(attr["id"])
        rprint("attribution:", (attr.get("result") or {}).get("recommendation"))
        # production intelligence: drift / failure clusters / coverage / recommendations
        rep = c.post("/intelligence/run", json={"sync": True, "recent_hours": 1, "baseline_hours": 24})
        rprint("intelligence:", rep["summary"] if "summary" in rep else {"alerts": len(rep["drift"].get("alerts", [])), "recommendations": len(rep["recommendations"])})
        # simulated human review of the judge (calibration): reviewers mostly agree with the verifier, with some noise
        ros = c.get("/rollouts", limit=60)
        ros = ros.get("items", ros) if isinstance(ros, dict) else ros
        c.post("/scores/batch", json=[{"name": "human_rating", "source": "human", "rollout_id": r["id"],
                                       "value": 1.0 if (r.get("total_reward", 0) > 0.5) != (rng.random() < 0.15) else 0.0} for r in ros])
        c.post("/orgs/current/webhooks", json={"url": "https://example.invalid/saphire-hook", "events": ["job.failed", "gate.decided", "intelligence.alert"],
                                               "description": "demo webhook (unreachable endpoint; shows retry/delivery log)"})
    rprint("[green]demo complete[/green]")


@app.command()
def bench(host: Optional[str] = typer.Option(None, help="running API to load-test (omit for offline benchmarks only)"), api_key: str = "dev-key",
          out: Optional[Path] = None, quick: bool = False):
    """Load / scale benchmarks with real numbers (rollout throughput per backend, persistence, ingest, API latency)."""
    from .bench import main as _bench

    _bench(host, api_key, out, quick)


@app.command()
def retention(days: Optional[int] = typer.Option(None, help="override SAPHIRE_RETENTION_DAYS / org settings"), dry_run: bool = False):
    """Delete traces/spans/rollouts older than the retention window (per org). Schedule via cron / K8s CronJob."""
    from .server import db as D
    from .server.retention import run_retention

    db = D.session()
    try:
        rprint(run_retention(db, days, dry_run=dry_run))
    finally:
        db.close()


@app.command("create-org")
def create_org_cmd(slug: str, name: Optional[str] = None, plan: str = "enterprise", owner_email: Optional[str] = None,
                   allowed_domain: Optional[str] = None):
    """Bootstrap an organization (and optionally its owner + SSO email domain) directly in the database."""
    from .server import db as D
    from .server.auth import create_api_key, upsert_user

    db = D.session()
    try:
        org = D.ensure_org(db, slug, name or slug, plan=plan)
        if allowed_domain:
            org.settings = {**(org.settings or {}), "allowed_email_domains": [allowed_domain]}
            db.commit()
        if owner_email:
            u = upsert_user(db, owner_email)
            if db.query(D.Membership).filter_by(org_id=org.id, user_id=u.id).one_or_none() is None:
                db.add(D.Membership(id=D.uid("mem"), org_id=org.id, user_id=u.id, role="owner"))
                db.commit()
        row, raw = create_api_key(db, org, "bootstrap admin key", role="admin", created_by="cli")
        rprint({"org": org.slug, "plan": org.plan, "admin_api_key": raw, "note": "shown once"})
    finally:
        db.close()


@app.command()
def migrate(revision: str = "head"):
    """Apply database migrations (Alembic). `saphire serve` also creates missing tables for dev, but production
    deployments should run this on upgrade."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent.parent / "migrations"))
    command.upgrade(cfg, revision)
    rprint(f"[green]migrated to {revision}[/green]")


@app.command()
def envs():
    """List built-in environments and their tools."""
    from .environments.base import get_environment, list_environments

    for n in list_environments():
        e = get_environment(n)
        rprint(f"[bold]{n}[/bold] – {e.description} ({len(e.tools)} tools)")


if __name__ == "__main__":
    app()
