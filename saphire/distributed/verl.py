"""verl integration (multi-node PPO/GRPO with vLLM/SGLang rollouts).

Two pieces make Saphire environments usable from verl without importing the whole platform into the trainer:

1. `export_verl_dataset(rows, path)` writes GRPO prompts (from `saphire.signals.generate.grpo_prompts`) as a parquet
   file in verl's expected schema:

       prompt        – chat messages list [{role, content}]
       data_source   – "saphire/<env>"
       reward_model  – {"style": "rule", "ground_truth": "<json: task, prefix_calls>"}
       extra_info    – {"task_id", "step_index", "env", "role"}

2. `compute_score(data_source, solution_str, ground_truth, extra_info)` is a verl *custom reward function*
   (`reward_model.reward_manager=naive` + `custom_reward_function.path=<this file>`,
   `custom_reward_function.name=compute_score`). It scores one sampled action by replaying the recorded prefix in a
   fresh Saphire environment — locally when `saphire` is importable in the trainer image, otherwise over HTTP against
   the environment server (`SAPHIRE_ENV_SERVER=http://host:8000`, `SAPHIRE_API_KEY`).

Launch example (see docs/DISTRIBUTED.md and deploy/verl_grpo.yaml):

    python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo \
        data.train_files=artifacts/verl/train.parquet data.val_files=artifacts/verl/val.parquet \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct actor_rollout_ref.rollout.name=vllm \
        custom_reward_function.path=saphire/distributed/verl.py custom_reward_function.name=compute_score \
        trainer.n_gpus_per_node=8 trainer.nnodes=2
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from ..sdk.llm import parse_assistant_text  # noqa: F401  (re-exported for reward parsing in trainers)


def _prompt_messages(prompt_text: str) -> list[dict[str, str]]:
    """Split Saphire's text chat format back into role/content messages."""
    out: list[dict[str, str]] = []
    role = None
    buf: list[str] = []
    for line in prompt_text.split("\n"):
        if line in ("<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"):
            if role is not None and (buf or role != "assistant"):
                out.append({"role": role, "content": "\n".join(buf).strip()})
            role, buf = line.strip("<|>"), []
        else:
            buf.append(line)
    if role is not None and buf and "".join(buf).strip():
        out.append({"role": role, "content": "\n".join(buf).strip()})
    return [m for m in out if m["content"]]


def to_verl_rows(rows: Iterable[dict[str, Any]], tasks_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        task = tasks_by_id.get(r["task_id"])
        task_d = task.model_dump() if hasattr(task, "model_dump") else (task or {"id": r["task_id"], "env_name": r["env"], "instruction": "", "expected": {"tools": r.get("expected_tools", [])}})
        out.append({
            "prompt": _prompt_messages(r["prompt"]),
            "data_source": f"saphire/{r['env']}",
            "reward_model": {"style": "rule", "ground_truth": json.dumps({"task": task_d, "prefix_calls": r.get("prefix_calls", [])})},
            "extra_info": {"task_id": r["task_id"], "step_index": r.get("step_index", 0), "env": r["env"], "role": r.get("role", "main")},
        })
    return out


def export_verl_dataset(rows: Iterable[dict[str, Any]], path: str | Path, tasks_by_id: dict[str, Any], val_fraction: float = 0.1) -> dict[str, str]:
    """Write train/val parquet files (falls back to JSONL when pyarrow is unavailable)."""
    data = to_verl_rows(rows, tasks_by_id)
    n_val = max(1, int(len(data) * val_fraction)) if len(data) > 1 else 0
    train, val = data[n_val:], data[:n_val]
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    files = {}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for name, part in (("train", train), ("val", val)):
            table = pa.Table.from_pylist([{**d, "prompt": json.dumps(d["prompt"]), "reward_model": json.dumps(d["reward_model"]),
                                           "extra_info": json.dumps(d["extra_info"])} for d in part] or
                                         [{"prompt": "[]", "data_source": "", "reward_model": "{}", "extra_info": "{}"}])
            pq.write_table(table, str(path / f"{name}.parquet"))
            files[name] = str(path / f"{name}.parquet")
    except ImportError:  # pragma: no cover
        for name, part in (("train", train), ("val", val)):
            p = path / f"{name}.jsonl"
            p.write_text("\n".join(json.dumps(d) for d in part))
            files[name] = str(p)
    files["n_train"], files["n_val"] = str(len(train)), str(len(val))
    return files


# ---------------------------------------------------------------------------
# verl custom reward function
# ---------------------------------------------------------------------------
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict[str, Any]] = None, **kw: Any) -> float:
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    task, prefix = gt["task"], gt.get("prefix_calls", [])
    env_name = data_source.split("/", 1)[1] if "/" in data_source else task.get("env_name")
    server = os.getenv("SAPHIRE_ENV_SERVER")
    if server:
        import httpx

        r = httpx.post(f"{server.rstrip('/')}/v1/env/reward", json={"env": env_name, "task": task, "prefix_calls": prefix, "completion": solution_str},
                       headers={"x-api-key": os.getenv("SAPHIRE_API_KEY", "dev-key")}, timeout=60)
        r.raise_for_status()
        return float(r.json()["reward"])
    from ..sdk.types import TaskSpec
    from ..training.trl_trainers import replay_reward

    t = TaskSpec.model_validate(task)
    return float(replay_reward(solution_str, env_name, t.id, prefix, t.expected.get("tools", []), tasks_by_id={t.id: t}))
