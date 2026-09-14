"""Weight-update trainers on top of HuggingFace TRL (optional `train` extra).

  sft(rows, ...)   – supervised fine-tuning on (prompt, completion) from successful trajectories
  dpo(rows, ...)   – direct preference optimisation on (prompt, chosen, rejected) pairs
  grpo(rows, ...)  – group-relative policy optimisation with an *environment-grounded* reward:
                     each sampled completion is parsed into an action, the recorded prefix of tool
                     calls is replayed in a fresh environment, and the action is scored by the
                     environment's verifier signal (tool correctness) plus format rewards.

All three run on CPU with a tiny model for CI (`saphire train sft --model sshleifer/tiny-gpt2`)
and scale to real models with LoRA on a GPU. They return the checkpoint path, which can be
used as an agent model via `hf:<path>`.

For large-scale multi-node RL (PPO/GRPO with vLLM/SGLang rollouts) see docs/TRAINING.md for
the verl / OpenRLHF / agent-lightning integration notes; Saphire's datasets are exported in a
format those frameworks consume directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..environments.base import get_environment
from ..sdk.llm import parse_assistant_text
from ..sdk.types import ToolCall

DEFAULT_TINY = "sshleifer/tiny-gpt2"


def _require():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("Weight-update training requires `pip install saphire[train]`") from e


def _peft_config(lora_r: int):
    if lora_r <= 0:
        return None
    from peft import LoraConfig

    return LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM")


def _fit_rows(rows: list[dict[str, Any]], tok, max_length: int, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Drop rows whose prompt + longest target would not fit (they would be fully masked)."""
    keep = []
    for r in rows:
        n = len(tok(r["prompt"])["input_ids"]) + max(len(tok(r[f])["input_ids"]) for f in fields if f in r)
        if n <= max_length:
            keep.append(r)
    return keep


def _load(model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model)
    return mdl, tok


def sft(rows: list[dict[str, Any]], output_dir: str | Path, model: str = DEFAULT_TINY, epochs: float = 1.0,
        lr: float = 5e-5, lora_r: int = 0, max_length: int = 1024, batch_size: int = 2, max_steps: int = -1) -> dict[str, Any]:
    _require()
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    mdl, tok = _load(model)
    rows = _fit_rows(rows, tok, max_length, ("completion",))
    if not rows:
        raise ValueError("no SFT rows fit within max_length; increase max_length or shorten prompts")
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    args = SFTConfig(output_dir=str(output_dir), num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=batch_size,
                     max_length=max_length, logging_steps=1, save_strategy="no", report_to=[], max_steps=max_steps,
                     completion_only_loss=True, use_cpu=not _has_cuda(), bf16=False, fp16=False)
    trainer = SFTTrainer(model=mdl, args=args, train_dataset=ds, processing_class=tok, peft_config=_peft_config(lora_r))
    out = trainer.train()
    path = _save(trainer, tok, output_dir, lora_r)
    return {"model": f"hf:{path}", "path": path, "train_loss": float(out.training_loss), "n_examples": len(rows), "algorithm": "sft"}


def dpo(rows: list[dict[str, Any]], output_dir: str | Path, model: str = DEFAULT_TINY, epochs: float = 1.0, lr: float = 5e-6,
        beta: float = 0.1, lora_r: int = 0, max_length: int = 1024, batch_size: int = 2, max_steps: int = -1) -> dict[str, Any]:
    _require()
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    mdl, tok = _load(model)
    rows = _fit_rows(rows, tok, max_length, ("chosen", "rejected"))
    if not rows:
        raise ValueError("no DPO pairs fit within max_length; increase max_length or shorten prompts")
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    args = DPOConfig(output_dir=str(output_dir), num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=batch_size,
                     beta=beta, max_length=max_length, logging_steps=1, save_strategy="no", report_to=[], max_steps=max_steps,
                     use_cpu=not _has_cuda(), bf16=False, fp16=False)
    trainer = DPOTrainer(model=mdl, args=args, train_dataset=ds, processing_class=tok, peft_config=_peft_config(lora_r))
    out = trainer.train()
    path = _save(trainer, tok, output_dir, lora_r)
    return {"model": f"hf:{path}", "path": path, "train_loss": float(out.training_loss), "n_pairs": len(rows), "algorithm": "dpo"}


# ---------------------------------------------------------------------------
# GRPO with environment-grounded reward
# ---------------------------------------------------------------------------
def replay_reward(completion: str, env_name: str, task_id: str, prefix_calls: list[dict], expected_tools: list[str],
                  tasks_by_id: Optional[dict[str, Any]] = None) -> float:
    """Score one sampled action: +1 if it calls an expected tool not yet called with valid args (the env
    executes it without error), -1 for a wrong tool, -0.5 for malformed output, small bonus for a final
    answer once all expected tools were called."""
    msg = parse_assistant_text(completion)
    called = [c["name"] for c in prefix_calls]
    remaining = [t for t in expected_tools if t not in called]
    if not msg.tool_calls:
        if not remaining and msg.content.strip():
            return 0.5  # correct to stop
        return -0.5 if not msg.content.strip() else -0.2
    tc = msg.tool_calls[0]
    if tc.name not in expected_tools:
        return -1.0
    if tc.name not in remaining:
        return -0.3  # repeated tool
    # execute in a fresh env after replaying the prefix to validate arguments
    env = get_environment(env_name)
    task = (tasks_by_id or {}).get(task_id)
    from ..sdk.types import TaskSpec

    state = env.reset(task or TaskSpec(env_name=env_name, instruction=""))
    for c in prefix_calls:
        env.tools.call(ToolCall(name=c["name"], arguments=c.get("arguments") or {}), state=state)
    res = env.tools.call(ToolCall(name=tc.name, arguments=tc.arguments), state=state)
    return 1.0 if res.error is None else 0.2


def grpo(rows: list[dict[str, Any]], output_dir: str | Path, model: str = DEFAULT_TINY, lr: float = 1e-5, num_generations: int = 4,
         max_completion_length: int = 48, lora_r: int = 0, max_steps: int = 5, batch_size: int = 4, beta: float = 0.0,
         max_prompt_length: int = 900) -> dict[str, Any]:
    _require()
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    mdl, tok = _load(model)
    rows = [r for r in rows if len(tok(r["prompt"])["input_ids"]) + max_completion_length <= max_prompt_length + max_completion_length]
    ds = Dataset.from_list([{"prompt": r["prompt"], "env": r["env"], "task_id": r["task_id"],
                             "prefix_calls": json.dumps(r.get("prefix_calls", [])), "expected_tools": json.dumps(r.get("expected_tools", []))}
                            for r in rows])

    def env_reward(completions, env, task_id, prefix_calls, expected_tools, **kw):
        return [replay_reward(c, e, t, json.loads(p), json.loads(x)) for c, e, t, p, x in zip(completions, env, task_id, prefix_calls, expected_tools)]

    def format_reward(completions, **kw):
        return [0.2 if "<tool_call>" in c and "</tool_call>" in c else 0.0 for c in completions]

    args = GRPOConfig(output_dir=str(output_dir), learning_rate=lr, num_generations=num_generations, per_device_train_batch_size=batch_size,
                      max_completion_length=max_completion_length, logging_steps=1,
                      save_strategy="no", report_to=[], max_steps=max_steps, beta=beta, use_cpu=not _has_cuda(), bf16=False, fp16=False)
    trainer = GRPOTrainer(model=mdl, reward_funcs=[env_reward, format_reward], args=args, train_dataset=ds, processing_class=tok,
                          peft_config=_peft_config(lora_r))
    out = trainer.train()
    path = _save(trainer, tok, output_dir, lora_r)
    rewards = [h.get("reward") for h in trainer.state.log_history if "reward" in h]
    return {"model": f"hf:{path}", "path": path, "train_loss": float(out.training_loss), "n_prompts": len(rows), "algorithm": "grpo",
            "reward_curve": rewards}


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _save(trainer, tok, output_dir, lora_r: int) -> str:
    path = Path(output_dir) / "final"
    path.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    if lora_r > 0 and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.save_pretrained(str(path))
    tok.save_pretrained(str(path))
    return str(path)
