# Production intelligence

`saphire.intelligence` closes the loop between production traffic and the training/evaluation stack. It reads the platform's
own records (rollouts, traces, scores, eval runs) and produces a report with executable recommendations; nothing in it needs
an external service.

## What a report contains

| Section | Source | Method |
|---|---|---|
| **Drift** | rollouts in a *recent* window vs a *baseline* window (per agent or all) | permutation tests on task success / tool-selection F1 / reward, error-rate and latency-ratio checks, Jensen–Shannon divergence of the tool-usage distribution, per-family regressions. Alerts: `regression`, `errors`, `inflation` (judge scores rising while verifiers fall), `tool_shift`, `family_regression`. `healthy` is true when there are no alerts. |
| **Failure clusters** | failed rollouts in the recent window | grouped by `(kind, role, tool, family)` where kind ∈ `wrong_tool`, `tool_error`, `missing_tool`, `incomplete`, `context_loss`; each cluster carries count, rate, baseline rate and `trend` (`new` / `growing` / `stable`) plus example rollout and task ids. |
| **Coverage** | production request patterns vs the tasks in eval datasets | share of production traffic whose tool pattern is not exercised by any eval task, and the top uncovered patterns. |
| **Task mining** | high-reward or human-approved rollouts / traces | turns real traffic into `TaskSpec`s (expected tools, order, entities) → a dataset with suite `mined`; human scores override verifier rewards. |
| **Recommendations** | everything above + the latest multi-agent attribution job | prioritised list; each item has `action = {kind, algorithm, params}` that `POST /v1/intelligence/apply` executes as-is (training run with `optimize_roles`, tool-description learning, reward-model refit, attribution job, task mining…). |

## API

```
POST /v1/intelligence/run          {sync?, agent_name?, recent_hours, baseline_hours, production_only}  → report (sync) or job
GET  /v1/intelligence/latest       most recent report for the project
GET  /v1/intelligence/reports      history;  GET /v1/intelligence/reports/{id}
POST /v1/intelligence/mine-tasks   {name, agent_name?, since_hours, min_reward, use_human_scores, production_only, max_tasks, split} → dataset
POST /v1/intelligence/apply        {recommendation, agent_id?, train_dataset_id?, eval_dataset_id?} → training run / attribution job / dataset
```

Reports can be scheduled (`POST /v1/intelligence/run` without `sync` enqueues an `intelligence` job; run it from a K8s CronJob or
`saphire worker`). Every unhealthy report fires the `intelligence.alert` webhook.

## Webhooks

`POST /v1/orgs/current/webhooks` `{url, events[], secret?, description}`. Events: `job.succeeded`, `job.failed`, `gate.decided`,
`intelligence.alert`, `quota.exceeded`, `test`. Payloads are signed with `X-Saphire-Signature: sha256=<HMAC-SHA256(secret, body)>`,
retried with back-off, and every attempt is stored (`GET /v1/orgs/current/webhooks/{id}/deliveries`).

## Judge calibration and tool statistics

* `GET /v1/signals/calibration?judge=judge_score|rm:<name>` — agreement, Cohen's κ, precision/recall/F1, best threshold and a
  reliability diagram of a judge (or reward model) against human/product scores on the same rollouts — the paper recipe for
  trusting a judge before using it as a training signal.
* `GET /v1/signals/tool-stats?agent_name=` — per-tool calls, error rate, success rate, latency and the confusion matrix of
  tools chosen instead of the expected one. The `tool_descriptions` training algorithm turns this into description hints
  (`AgentConfig.tool_description_overrides`).

## Multi-agent attribution

`POST /v1/attribution` `{agent_id, dataset_id, reference_model?, degraded_model?, roles?, shapley, n_permutations}` runs
counterfactual **role ablation** (headroom = success with the role upgraded to `reference_model` minus baseline; criticality =
loss when the role is degraded) and **Shapley values** of upgrading each role (Monte-Carlo permutations, or exact for ≤ 6 roles).
`GET /v1/evals/{id}/attribution` returns cheaper per-eval **blame** (which role made the first mistake) and **advantage credit**
(per-role reward advantage vs the task mean). Both feed `recommendations[].action.params.optimize_roles`.

Dashboard pages: **Intelligence**, **Attribution**, **Signals**, **Webhooks**.
