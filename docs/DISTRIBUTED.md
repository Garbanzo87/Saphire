# Distributed rollouts and RL

Rollout collection and evaluation are embarrassingly parallel; weight-update RL is not. Saphire separates the two:

## 1. Distributed rollouts (bundled)

`saphire.distributed.RolloutEngine` shards (task, trial) pairs across workers and returns verified rollouts + records
that aggregate exactly like the local runner:

| backend | when | how |
|---|---|---|
| `thread` | hosted LLMs (I/O bound) | thread pool in-process |
| `process` | CPU-bound local policies / verifiers | `ProcessPoolExecutor` (spawn) |
| `ray` | multi-node | one Ray actor per worker holding its environments and agent; `RAY_ADDRESS` or `ray_address=` for a cluster |

```python
from saphire.distributed import evaluate_distributed
res = evaluate_distributed(config, tasks, k=3, backend="ray", workers=64)
```

* CLI: `saphire eval --suite full --distributed ray --workers 64`
* Online loop / training job: `OnlineLoop(..., rollout_backend="ray", rollout_workers=64)` or `params.rollout_backend` on
  `POST /v1/training` — collection *and* held-out evaluation run on the cluster; learner updates stay on the worker.
* Agents are rebuilt from the serialised `AgentConfig` on each worker, so artifacts must be reachable from every node:
  set `SAPHIRE_ARTIFACT_STORE=s3://bucket/prefix` (see docs/DEPLOYMENT.md) or mount a shared volume.
* Measured on 2 vCPU (docs/BENCHMARKS.md): ~10k rollouts/min with the mock policy; with hosted models throughput is
  provider-latency bound and scales ~linearly with workers.

## 2. Environment / reward server (bundled)

External trainers do not need Saphire in their image. `saphire env-server` (or the same routes under `/v1/env` on the
API) exposes:

* stateless `POST /v1/env/reward` and `/reward/batch` — environment-grounded reward for sampled actions (GRPO groups),
* `POST /v1/env/verify` — replay a full trajectory and run the verifier,
* stateful sessions `POST /v1/env/sessions` → `/step` → `/user_turn` → `/finish` for step-by-step episodes
  (pin to one replica — the Helm service uses ClientIP affinity).

## 3. verl (multi-node PPO/GRPO with vLLM/SGLang)

1. Generate prompts: run the `signals` training job or `saphire.signals.generate.grpo_prompts`, then
   `saphire.distributed.verl.export_verl_dataset(rows, "artifacts/verl", tasks_by_id)` → `train.parquet`/`val.parquet`
   in verl's schema (`prompt` messages, `data_source=saphire/<env>`, `reward_model.ground_truth` = task + replayable prefix).
2. Point verl at the custom reward function `saphire/distributed/verl.py:compute_score`. It replays the prefix and scores the
   sampled action in-process, or over HTTP when `SAPHIRE_ENV_SERVER` is set.
3. Launch with `deploy/verl_grpo.yaml` (2×8 GPUs example). Not executed in CI — needs GPUs.

## 4. OpenRLHF / SkyRL-style agent loops

`saphire/distributed/openrlhf.py` implements the async `AgentInstance.reset/step` protocol: the trainer samples actions,
Saphire runs tools, simulated user turns and the verifier (in-process or via the env server). Export prompts with
`export_openrlhf_prompts(tasks, "prompts.jsonl")` and pass `--agent_func_path saphire/distributed/openrlhf.py`.

## 5. Single-node weight updates (bundled, tested)

`saphire.training.trl_trainers` — SFT / DPO / GRPO with TRL + LoRA, CPU smoke path in CI. Fine up to ~7B with LoRA on one
node; beyond that use §3.

## What is and is not verified here

Verified in this repo's tests: thread/process/Ray engines, env/reward server endpoints, verl parquet export and `compute_score`,
the OpenRLHF `AgentInstance` protocol in-process, TRL trainers on a tiny model. Not executed (no GPUs in CI): an actual verl or
OpenRLHF run. The integration surface is small (one reward function / one agent class) and is exercised by the tests above.
