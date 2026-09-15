# Pilot playbook — producing real production evidence

Saphire has no customer evidence yet and this document does not pretend otherwise. It is the procedure for turning the
first deployments into publishable, reproducible evidence (the kind Metis claims but has not published).

## Week 0 — baseline (2–3 days)
1. Instrument the existing agent with `saphire.sdk.tracing` (or attach OpenInference instrumentors). No behaviour change.
2. Build the environment: connectors (`SQLConnector`, `OpenAPIConnector`, MCP) or a `FileDataConnector` from a data export;
   50–200 tasks with ground truth from real tickets/transcripts (`DataEnvironment` + `TaskVerifier`, plus an LLM judge
   calibrated against 50 human-labelled transcripts — report agreement).
3. Freeze a held-out dataset (`split=eval`, fixed seed). Run the baseline eval with `k=3`: record `task_success`,
   `pass_hat_3`, `tool_selection_f1`, `context_preservation`, p95 latency, tokens/task.

## Weeks 1–3 — learn
4. Online loop on the train split (router + exemplars + prompt optimisation; SFT/DPO/GRPO when a self-hosted model exists).
   Every iteration writes a metric point → improvement-over-time chart.
5. Gate each candidate against the deployed version (`GatePolicy`, non-regression p < 0.1) before promotion.
6. Shadow / A/B on live traffic (`/v1/experiments`) with a business outcome (resolution, CSAT, handle time).

## Week 4 — evidence package
Report, from the platform's own records (all exportable):
* baseline vs final held-out metrics with 95% CIs and permutation-test p-values (`/v1/evals/{a}/compare/{b}`),
* pass^k reliability and consistency,
* A/B result with n, delta, CI, p-value,
* throughput/latency/cost per task before and after,
* the audit trail of every promotion.
Fill `docs/CASE_STUDY_TEMPLATE.md`. Only numbers produced by the platform go in; nothing is extrapolated.

## Scale checklist for the first production tenant
Postgres + ≥2 workers (KEDA on pending jobs), `SAPHIRE_ARTIFACT_STORE=s3://…`, OIDC SSO, org API keys (root key rotated),
retention configured, `saphire bench` run on the production shape and its numbers attached to the evidence package.
