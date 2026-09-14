# API (summary — full OpenAPI at `/docs`)

All endpoints under `/v1`, header `x-api-key`, optional `?project=<name>` (auto-created).

| Method & path | Purpose |
|---|---|
| `GET /projects`, `POST /projects` | projects |
| `GET /environments`, `GET /environments/{name}/tools`, `GET /suites` | built-in environments and suites |
| `GET/POST /agents`, `GET /agents/{id}`, `POST /agents/{id}/promote`, `DELETE /agents/{id}` | agent versions |
| `GET/POST /datasets`, `GET /datasets/{id}`, `POST /datasets/{id}/tasks` | task datasets (from suite or explicit tasks) |
| `POST /traces/ingest`, `GET /traces`, `GET /traces/{id}` | OTel spans in; traces with spans + scores out |
| `POST /rollouts`, `GET /rollouts`, `GET /rollouts/{id}` | BYO rollouts; stored rollouts with records and payloads |
| `POST /scores`, `POST /scores/batch`, `GET /scores`, `GET /scores/summary` | signals |
| `POST /evals`, `GET /evals`, `GET /evals/{id}`, `GET /evals/{a}/compare/{b}` | evaluation runs (async job), comparison |
| `POST /training`, `GET /training`, `GET /training/{id}` | training runs: `online, router, exemplars, prompt_opt, signals, sft, dpo, grpo` |
| `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` | job queue |
| `GET /deployments`, `GET /deployments/current`, `POST /deployments/{id}/promote` | gate decisions and promotion |
| `POST /experiments`, `GET /experiments`, `GET /experiments/{id}`, `GET /experiments/{id}/assign?unit=`, `POST /experiments/{id}/outcomes`, `POST /experiments/{id}/stop` | live A/B |
| `GET /metrics/timeseries?agent_name=&name=`, `GET /metrics/overview` | improvement over time, dashboard summary |

Python: `saphire.sdk.SaphireClient` wraps the common calls (`create_agent`, `create_dataset`, `run_eval(wait=True)`, `train(wait=True)`,
`log_rollout`, `score`, `assign_variant`, `record_outcome`).
