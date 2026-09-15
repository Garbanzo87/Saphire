# Saphire vs. Metis (withmetis.ai) — verified comparison and capability gaps

Research date: 2026-09-14 (scorecard updated 2026-09-15 for Saphire 0.3.0). Sources: [withmetis.ai](https://www.withmetis.ai/), [docs.withmetis.ai](https://docs.withmetis.ai/),
PyPI [`mantisdk`](https://pypi.org/project/mantisdk/) (0.2.4, MIT), GitHub [metis-mantis](https://github.com/metis-mantis)
(`metis-router`, Apache-2.0), arXiv [2603.03565](https://arxiv.org/abs/2603.03565), YC / PitchBook / Fenwick announcements.

## What Metis publicly is

* YC S25 applied-research lab; **acquired by DoorDash (announced 17–18 March 2026)**. Landing page is a single page; `/docs`, `/blog`, `/pricing` 404.
* Marketing claims for the product "Insight" (a.k.a. "Mantis" platform): *learns from your first-party data, tools and environments;
  generates training signals and runs offline/online RL; evaluates agents on real tasks so they pick the right tool, preserve context and
  complete long multi-step workflows with higher reliability and throughput.*
* **Publicly verifiable artifacts**
  * `mantisdk` — a rebranded fork of Microsoft's MIT-licensed [agent-lightning](https://github.com/microsoft/agent-lightning)
    (98 files keep the Microsoft copyright): Trainer / Algorithm / Runner / Store / LLMProxy, `verl` PPO/GRPO hookup, APO, GEPA,
    single-pass "Evals", A/B `experiments`, OTel `tracing` exporting to "Insight" (Langfuse-shaped: `pk-lf-`/`sk-lf-` keys),
    Docker/BullMQ agent orchestration with `agent.yml`/`task.yml` manifests. The platform UI/server it targets is closed.
  * `docs.withmetis.ai` documents **only the tracing SDK** (OpenInference instrumentors, scores API, MCP server for trace queries).
  * `metis-router` — MCP router with embedding-based tool/server selection over "1,000+ MCP servers".
  * Paper 2603.03565 — LLM-as-judge with boolean checks over trace evidence + GEPA prompt optimisation on DoorDash's grocery assistant
    (judge agreement 84.1→91.4 %, MAMUT-GEPA 84.7 % vs sub-agent GEPA 77.1 %). **No RL results published.**
* **Claimed only** (no public artifact or numbers): RL on customer data, reward modelling from first-party data, reliability/throughput gains,
  "autonomous post-training agent", frontier-lab / Fortune-500 customers.

## Feature matrix

| Capability | Metis (verifiable) | Saphire | Notes |
|---|---|---|---|
| OTel tracing SDK, decorators, spans for agent/llm/tool | ✅ `mantisdk.tracing` | ✅ `saphire.sdk.tracing` | Same span-attribute conventions (`input.value`, `output.value`, `llm.token_count.*`); OpenInference instrumentors attach to the same provider |
| Scores / signals API on traces | ✅ | ✅ `POST /v1/scores`, `/scores/batch` | plus rollout-level scores with provenance |
| Trace UI (timeline, tokens, tools) | ✅ (closed, Langfuse-based) | ✅ open dashboard (waterfall, attributes, scores) | |
| Framework-agnostic rollout capture | ✅ (agent-lightning) | ✅ `RolloutRecorder` + `POST /v1/rollouts` | |
| Environments / tools from first-party data | ⚠️ manifests only; no public env library | ✅ SQL, OpenAPI, CSV/JSON, MCP connectors + `DataEnvironment`; 2 built-in simulated envs | Metis claims this; Saphire ships it |
| Ground-truth verifiers on final world state | ❌ not public | ✅ per-environment verifiers, `TaskVerifier` for custom envs | |
| LLM-as-judge with boolean evidence checks | ✅ (paper) | ✅ `RubricJudge` (+ offline heuristic judge) | |
| Pairwise / relative judging | ❌ | ✅ `PairwiseJudge` | RULER-style |
| Training-signal generation (SFT / DPO / GRPO / router datasets) | ⚠️ triplet export via agent-lightning | ✅ `saphire.signals.generate` + `signals` job exporting JSONL | |
| Prompt optimisation (GEPA/APO) | ✅ via `gepa` | ✅ reflective Pareto optimiser (`optimize_prompt`) | Saphire's is self-contained; can call any LiteLLM model for reflection |
| Tool routing for large catalogues | ✅ metis-router (embedding, MCP) | ✅ learnable `ToolRouter` trained from rewards | Saphire's router *learns* from outcomes; Metis's is zero-shot semantic |
| In-context exemplar learning | ❌ | ✅ `ExemplarStore` | |
| Online learning loop with versioned agents | ⚠️ claimed | ✅ `OnlineLoop` + `training` job → agent versions + metric points | |
| SFT / DPO on trajectories | ✅ (agent-lightning SFT) | ✅ TRL, LoRA, CPU smoke path in CI | |
| RL with environment reward (GRPO/PPO) | ✅ thin `verl` wrapper, GPU only | ✅ TRL GRPO with replay-grounded reward (single node) + verl/OpenRLHF integrations (see Distributed RL row) | |
| Evaluation on tool selection / context / long workflows | ⚠️ claimed | ✅ dedicated suites + metrics | |
| Reliability metrics pass@k / pass^k, CIs | ❌ | ✅ | |
| Throughput / latency / tokens | ✅ token/latency in trace UI | ✅ per eval run + per rollout | |
| Improvement over time | ⚠️ | ✅ metric time series per agent version + dashboard | |
| Deployment gates with statistical non-regression | ❌ | ✅ `GatePolicy`, auto-promote | |
| A/B experiments on live traffic | ✅ SDK-first `experiments` | ✅ sticky assignment + outcomes + p-values | |
| Job orchestration | ✅ BullMQ/Docker/Nomad/KubeRay | ✅ DB-backed queue + `saphire worker`; K8s manifests | simpler; no per-job containers |
| Agent manifests / CLI | ✅ `mantis register/run/...` | ✅ `saphire serve/worker/eval/train/demo` + `AgentConfig` JSON | |
| MCP server to query traces from IDEs | ✅ `@mantisai/mcp` | ❌ | gap (easy: FastMCP over `/v1/traces`) |
| Multi-agent selective optimisation | ✅ (agent-lightning) | ✅ `AgentSystem` roles, per-role credit, `optimize_roles` freezes the rest; joint multi-prompt optimisation | tested end-to-end |
| Multi-agent attribution (which role to optimise) | ⚠️ implicit in training | ✅ blame, advantage credit, counterfactual role ablation (headroom / criticality), Shapley values, `POST /v1/attribution` | recommendation feeds selective optimisation |
| Production intelligence (drift, failure clusters, coverage, task mining, recommendations) | ⚠️ claimed ("autonomous post-training agent") | ✅ `saphire.intelligence`, `/v1/intelligence/*`, one-click apply, `intelligence.alert` webhook | see docs/INTELLIGENCE.md |
| Reward model from first-party signals | ⚠️ claimed | ✅ `RewardModel` on trajectory features, `rm:` judge, calibration (κ, reliability diagram) against human scores | |
| Tool learning from outcomes | ✅ semantic router | ✅ reward-trained router with reliability prior + learned tool-description hints from confusion statistics | |
| Webhooks / Prometheus / migrations | ✅ (closed) | ✅ HMAC-signed webhooks with delivery log, `/metrics`, Alembic migrations, Postgres CI job | |
| Distributed RL | ✅ verl wrapper | ✅ Ray/process/thread rollout engine, env/reward server, verl dataset + reward fn, OpenRLHF agent | verl/OpenRLHF runs themselves need GPUs (not in CI) |
| Multi-tenant SaaS (orgs, keys, quotas, metering) | ✅ (closed) | ✅ orgs, hashed org API keys, plans/quotas (402), rate limits (429), usage metering | self-hosted; no billing integration |
| SSO / RBAC / audit | ✅ (closed) | ✅ OIDC login + JWT sessions, JIT provisioning by email domain, SCIM 2.0 user provisioning, 4 roles, per-request audit log, API-key IP allowlists | |
| Production evidence | claimed | ❌ none — pilot playbook + case-study template + benchmark harness only | must be earned |
| Real-world scale | production | ⚠️ measured platform overhead (docs/BENCHMARKS.md), bulk ingest, retention, S3 artifacts, HPA/KEDA, Helm | no production tenant yet |

## Capability gaps (honest list)

1. **Scale-out RL.** Rollout collection and evaluation are distributed (Ray); environment rewards are served over HTTP;
   verl/OpenRLHF integrations exist (dataset export, reward function, agent loop) but the multi-node PPO/GRPO runs themselves
   are not executed in this repo's CI (no GPUs). Bundled weight updates are single-node TRL (SFT/DPO/GRPO).
2. **Live-system connectors are generic.** SQL/OpenAPI/MCP/CSV connectors cover most enterprise systems, but there are no
   turnkey Salesforce/Zendesk/Slack adapters and no PII redaction layer for traces (the tracing attribute serialiser is the hook).
3. **Judges need a hosted model to be meaningful.** The offline heuristic judge exists so pipelines run in CI; production
   judging should point `judge_model` at a real model and calibrate it against human labels (the paper's recipe).
4. **Mock policy is a test double.** Learning gains shown by `saphire demo` come from the router, exemplars and prompt rules
   acting on a scripted weak policy; with real models the same mechanisms apply but gains must be measured per deployment.
5. **Enterprise surface is complete but young.** Orgs, hashed API keys (with IP allowlists), OIDC SSO, SCIM, RBAC, audit, quotas,
   metering, webhooks, Prometheus metrics, migrations and retention are implemented and tested; per-project roles,
   customer-managed keys, billing integration and compliance evidence (SOC 2) are not.
6. **No per-job sandboxing.** Jobs run in worker processes, not isolated containers (Metis uses Docker/BullMQ).
7. **Production evidence is zero.** No customer has run this. `docs/PILOT_PLAYBOOK.md` defines how the first deployment
   produces reproducible evidence from the platform's own records; `docs/BENCHMARKS.md` holds measured platform overhead only.
8. **Trace MCP server and a prompt-template registry** are not implemented.

## Scorecard (the 18 dimensions used in the user's Metis-vs-Saphire review)

Scores are the reviewer's Metis score and Saphire's self-assessed score after 0.3.0, with what changed. "Proven" means
demonstrated by tests/benchmarks in this repo, not by customers.

| Dimension | Metis | Saphire 0.2 | Saphire 0.3 | What closed the gap |
|---|---|---|---|---|
| Core architecture | 9.5 | 9 | 9.5 | intelligence layer + attribution wired into the same job queue / artifact / versioning model |
| Tracing & observability | 9 | 9 | 9 | Prometheus `/metrics`, DB span exporter |
| Environment grounding | 9 | 9 | 9.5 | task mining from production traces, coverage gaps, `SaphireGymEnv` |
| Reward / signal generation | 9 | 8.5 | 9 | `RewardModel` (`rm:` judge), judge calibration (κ, reliability), human-score overrides |
| Evaluation | 9.5 | 9 | 9.5 | per-eval blame / advantage credit, drift tests, failure clustering |
| Prompt optimisation | 10 | 8 | 9.5 | v2: per-task Pareto front, reflect + merge, joint multi-role optimisation, rollout budget |
| Tool learning | 9 | 8.5 | 9.5 | reliability prior in the router, learned tool-description hints from confusion stats |
| Continual-learning orchestration | 10 | 9 | 9.5 | recommendations → one-click training runs, scheduled intelligence jobs, webhooks |
| SFT / DPO / GRPO | 9 | 9 | 9 | unchanged (TRL, CPU smoke in CI) |
| Distributed RL | 10 | 8.5 | 9 | gym adapter, verl/OpenRLHF integrations; multi-node runs still not executed in CI (GPUs) |
| Multi-agent attribution | 10 | 5.5 | 9.5 | blame, advantage credit, role ablation, Shapley, dashboard |
| Deployment safety | 9 | 9 | 9 | unchanged (gates, permissions, auto-promote requires deploy right) |
| Platform backend | 9.5 | 8.5 | 9.5 | Alembic migrations, Postgres CI, `/metrics`, webhooks, IP allowlists, race-safe metering |
| Dashboard | 9 | 8.5 | 9 | Intelligence, Attribution, Signals, Webhooks pages; eval blame panel |
| Enterprise layer | 9.5 | 8 | 9 | SCIM 2.0, IP allowlists, webhooks; billing / SOC 2 evidence still absent |
| Production scalability | 10 | 6.5 | 7.5 | measured benchmarks + Postgres CI + autoscaling; unproven at a real tenant's scale |
| Production intelligence | 10 | 2 | 9 | `saphire.intelligence` (drift, clusters, coverage, mining, recommendations, apply) |
| Enterprise customer validation | 10 | 0 | **0** | cannot be built in code — see docs/PILOT_PLAYBOOK.md |

## External dependencies (all permissive licences)

FastAPI/Starlette (MIT), SQLAlchemy (MIT), Pydantic (MIT), OpenTelemetry (Apache-2.0), httpx (BSD), NumPy (BSD), Typer/Rich (MIT),
LiteLLM (MIT), MCP Python SDK (MIT), PyTorch (BSD), transformers/TRL/PEFT/datasets/accelerate (Apache-2.0), Next.js (MIT),
Recharts (MIT), SWR (MIT), Tailwind (MIT), Postgres (PostgreSQL licence). Optional heavy backends referenced in docs:
verl (Apache-2.0), OpenRLHF (Apache-2.0), agent-lightning (MIT), SkyRL (Apache-2.0), Inspect AI (MIT), DeepEval (Apache-2.0).
