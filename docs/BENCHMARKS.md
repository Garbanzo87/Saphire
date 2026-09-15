# Benchmarks

Measured 2026-09-15 10:31:44 UTC on 2 vCPU, Linux-6.18.44-fc-v33-x86_64-with-glibc2.39, Python 3.11.15. Numbers are from the mock policy (no LLM latency) and therefore measure the *platform overhead* — tracing, verification, persistence, aggregation — not model speed. With hosted models, throughput is bounded by provider latency and scales with concurrency/workers.

## Rollout throughput (224 rollouts, support_desk + data_ops, k trials)

| backend | wall s | rollouts/min | task_success |
|---|---|---|---|
| local | 2.02 | 6708 | 0.598 |
| thread | 1.32 | 10555 | 0.594 |
| process | 1.62 | 8486 | 0.558 |
| ray | 5.95 | 7809 | 0.589 |

Multi-agent system (orchestrator + 4 specialists, 9.0 steps/rollout): 9519 rollouts/min local.

## Persistence (sqlite (WAL))

* `persist_rollout` (rollout + scores + metering): **265 rollouts/s**
* paged rollout list query (offset 500, limit 50): p50 2.52 ms, p95 3.21 ms

## Trace ingest (HTTP, 8 concurrent clients, 20 spans/request)

* **1780 spans/s** (89.0 req/s), p50 77.3 ms, p95 196.6 ms, errors 0

## API read latency (populated DB)

| endpoint | p50 ms | p95 ms |
|---|---|---|
| `/v1/traces` | 8.9 | 10.3 |
| `/v1/metrics/overview` | 7.6 | 9.1 |
| `/v1/agents` | 5.0 | 5.9 |
| `/v1/traces/{id}` | 5.8 | 6.7 |

Reproduce: `saphire serve --inline-jobs &` then `saphire bench --host http://localhost:8000 --out docs/benchmarks.json`.
