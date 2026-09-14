# Evaluation

```bash
saphire eval --suite full --model openai/gpt-4o-mini --k 3 --judge openai/gpt-4o-mini --concurrency 8
# API: POST /v1/evals {"agent_id": ..., "dataset_id": ..., "k": 3, "judge_model": "openai/gpt-4o-mini", "gate": true, "auto_promote": false}
```

## Suites (`saphire.evaluation.suites`)

| Suite | Families | Measures |
|---|---|---|
| `tool_selection` | lookup, cancel_email, refund_resolve, kb_email, metric_alert, report_post | picking the right tool among 30 with 16 look-alike distractors; argument correctness via tool errors |
| `context_preservation` | multiturn (info revealed over 3 simulated user turns), find_list (id passed from tool output to next call) | `context_preservation` reward: later calls reuse the right identifiers |
| `long_horizon` | damage_workflow (6 ordered steps), quality_schedule (3 ordered steps) | ordered completion verified on final state, step efficiency |
| `cross_env` / `full` | everything across `support_desk` + `data_ops` | generalisation, per-env breakdowns |

Datasets are built from suites with a seed (`POST /v1/datasets {"suite": "full", "n_per_env": 28, "seed": 200}`) or supplied as
explicit `TaskSpec`s (your own tasks against your own environment).

## Metrics (per eval run, per family/env/difficulty, per rollout)

* `task_success` (binary, from verifier) with bootstrap CI; `pass_at_1`; with `k>1`: `pass_at_k`, `pass_hat_k` (all k trials succeed —
  the τ-bench reliability estimator), `consistency`.
* `tool_selection_f1`, `step_efficiency` (expected calls / actual calls), `context_preservation`, `tool_error_free`, `judge_score`.
* `error_rate` (infrastructure errors / timeouts), `latency_ms_p50/p95`, `tokens_per_task`, `steps_mean`, `throughput_tasks_per_min`.

## Comparing versions and gating deployments

* `GET /v1/evals/{baseline}/compare/{candidate}` → per-metric delta, permutation-test p-value, candidate CI.
* `GatePolicy` (`saphire.evaluation.gates`): `min_metrics`, `max_metrics`, `non_regression_metrics` with `max_regression` and
  `p_value`, optional `require_significant_improvement`. Passing an eval with `gate=true` records a `Deployment`; `auto_promote=true`
  promotes on pass (previous deployed version → `retired`).
* Every succeeded eval and every online-loop iteration writes `MetricPoint`s → `GET /v1/metrics/timeseries?agent_name=…&name=task_success`
  → the dashboard's improvement-over-time chart.

## Live A/B experiments

`POST /v1/experiments` with weighted variants (agent versions) → the SDK calls `GET /experiments/{id}/assign?unit=<user>` (sticky hash) and
reports `POST /experiments/{id}/outcomes`; results include per-variant mean, CI, delta and p-value.

## Judges

`RubricJudge(model)` asks a model for JSON `{check: {pass, evidence}}` over the transcript (boolean checks over concrete trace evidence —
the recipe reported to lift judge/human agreement in the Metis–DoorDash paper). With `model="mock"` a deterministic heuristic judge is used
so pipelines and CI run offline. Custom rubrics: `RubricJudge(model, rubric=[("id", "question"), ...])`.
