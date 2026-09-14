# Architecture

```
                         ┌─────────────────────────────── control plane ────────────────────────────────┐
                         │  FastAPI  (saphire.server)                                                    │
  agent process          │   /v1/traces/ingest  /v1/rollouts  /v1/scores   ← signals in                 │
 ┌─────────────────┐     │   /v1/agents  /v1/datasets  /v1/evals  /v1/training  /v1/deployments         │
 │ saphire.sdk     │ ──▶ │   /v1/experiments  /v1/metrics  /v1/jobs                                      │
 │  tracing        │     │                                                                               │
 │  ToolAgent /    │ ◀── │  SQLAlchemy models (SQLite | Postgres): projects, agents(versions), datasets,  │
 │  RolloutRecorder│     │  traces, spans, rollouts, scores, jobs, eval_runs, training_runs, experiments,│
 │  ToolRouter     │     │  metric_points, deployments                                                   │
 │  ExemplarStore  │     └───────────────────────────────────────────────────────────────────────────────┘
 │  SaphireClient  │                       │ jobs table (pending → running → succeeded/failed)
 └─────────────────┘                       ▼
                         ┌──────────────── workers (saphire worker ×N) ───────────────────────────────────┐
                         │  eval job:      evaluate(agent, tasks, k, judge) → rollouts + spans + metrics  │
                         │                 → optional GatePolicy vs deployed baseline → promote           │
                         │  training job:  online | router | exemplars | prompt_opt | signals | sft|dpo|grpo│
                         │                 → artifacts/<project>/<agent>/<run>/… → new agent version     │
                         └────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                         ┌──────────────── dashboard (Next.js static export) ─────────────────────────────┐
                         │ overview · agents · traces · rollouts · evals · training · deployments · A/B  │
                         └────────────────────────────────────────────────────────────────────────────────┘
```

## Data model

* `Rollout` (SDK, `saphire/sdk/types.py`) is the canonical trajectory: `steps[]` each with `prompt_messages`, `response`,
  `tool_results`, `usage`, `exposed_tools`; `rewards[]` with `name/value/source/step_index/rationale`; `trace_id`.
  Stored verbatim as JSON in `rollouts.payload` so any learner can be re-run on historical data.
* `RolloutRecord` (evaluation) is the flattened per-rollout metrics row; `EvalRun` stores aggregates and breakdowns.
* `Agent` rows are immutable **versions** (`name` + `version`), with `parent_id` lineage and `origin`
  (`manual|online|router|exemplars|prompt_opt|sft|dpo|grpo`). `status ∈ {candidate, deployed, retired}`.
* `MetricPoint` is the time series behind "improvement over time" (one row per metric per eval/iteration).

## Request → rollout → signal → update flow

1. `ToolAgent.run(task, env)`: system prompt (+ retrieved exemplars) → loop: router ranks catalogue → LLM → tool calls executed
   against the per-rollout `EnvState` → simulated user follow-ups → final answer. Every step is an OTel span.
2. `env.verify(task, rollout, state)` inspects **final state and log** → rewards. Optional `RubricJudge` adds `judge_score`.
3. `signals.generate` converts rollouts to learner datasets; `training.learners` / `prompt_opt` / `trl_trainers` produce artifacts.
4. `OnlineLoop.step()` = collect batch → update learners → write `agent.json` → evaluate on the fixed held-out set → record history.
5. An eval with `gate=true` compares against the deployed version's latest eval on the same dataset and can `auto_promote`.

## Concurrency and scaling

* API is stateless; workers claim jobs with an atomic `UPDATE … WHERE status='pending'`; run as many as needed.
* `evaluate(concurrency=N)` runs rollouts in a thread pool (I/O-bound LLM calls). Throughput is reported as tasks/min.
* SQLite (WAL) is fine for a single node; use Postgres (`SAPHIRE_DATABASE_URL=postgresql+psycopg://…`) for multi-worker deployments.
* Artifacts live on a shared filesystem (`SAPHIRE_ARTIFACTS_DIR`); on Kubernetes use a RWX PVC or an object-store mount.

## Extension points

| Want to… | Do |
|---|---|
| Add an environment | subclass `Environment` (`build_tools`, `seed_data`, `generate_tasks`, `verify`) and `@register_environment` |
| Add a signal source | `POST /v1/scores` or a `judge` callable `(task, rollout) -> list[Reward]` |
| Add a learner | write `fn(rollouts, cfg, out_dir) -> {artifact/model}` and wire it into `OnlineLoop(weight_update=…)` or a `jobs.py` algorithm |
| Add an LLM provider | implement `LLM.complete(messages, tools, temperature)`; register in `get_llm` |
| Add a metric | emit a trajectory-level `Reward` with a new `name`; add it to `TRAJ_METRICS` to aggregate |
