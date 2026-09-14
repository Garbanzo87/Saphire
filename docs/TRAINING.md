# Training

Saphire treats "training" broadly: anything that turns rewards into a better next agent version.

## 1. Signals

`saphire.signals.generate` turns verified rollouts into datasets (also exported as JSONL by the `signals` training job):

| Function | Rows | Used by |
|---|---|---|
| `router_examples` | `(query, tool, weight)`; +1 from successful trajectories, ±0.5 from step-level `tool_correct` | `ToolRouter.fit` |
| `sft_examples` | `{prompt, completion}` for every step of a successful rollout; prompt = chat rendered as text incl. the tools that were exposed | `trl_trainers.sft` |
| `preference_pairs` | `{prompt, chosen, rejected}` — first divergent step between a successful and a failed rollout of the same task | `trl_trainers.dpo` |
| `grpo_prompts` | `{prompt, task_id, env, step_index, prefix_calls, expected_tools}` — enough to *replay* the prefix and score any new action | `trl_trainers.grpo` |
| `step_rewards` | flat transition table (action, step reward, trajectory reward, exposed tools) | external RL frameworks |

Rewards come from three places: environment **verifiers** (ground truth on final state), **judges** (`RubricJudge`: boolean checks
with evidence, `PairwiseJudge`), and **humans/products** (`POST /v1/scores`).

## 2. Online learners (seconds on CPU)

```bash
saphire train online --model "mock:error=0.3" --iterations 5      # local
# or via API: POST /v1/training {"agent_id": ..., "algorithm": "online",
#   "params": {"train_dataset_id": ..., "eval_dataset_id": ..., "iterations": 5, "batch_size": 24, "learn_prompt_every": 2}}
```

Each iteration: collect a batch of rollouts with the current version → verify → retrain the **ToolRouter** on the replay buffer →
add successful trajectories to the **ExemplarStore** → every N iterations run reflective **prompt optimisation** → write
`artifacts/<agent>/<version>/{agent.json, router.json/npz, exemplars.json, eval.json}` → evaluate on the fixed held-out dataset.
Optionally `weight_update_every` / `weight_algorithm=sft|dpo` triggers a TRL update on the buffer.

## 3. Weight updates with TRL (`pip install saphire[train]`)

```bash
saphire train sft  --base-model sshleifer/tiny-gpt2 --max-steps 5      # CPU smoke
saphire train grpo --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 200   # GPU recommended
# API: POST /v1/training {"agent_id": ..., "algorithm": "grpo", "params": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "lora_r": 16, "max_steps": 300}}
```

* **SFT** – completion-only loss on successful steps. **DPO** – β=0.1 default. **GRPO** – reward functions:
  `env_reward` (parse `<tool_call>`, replay `prefix_calls` in a fresh environment, execute the sampled action: +1 expected tool & valid
  args, +0.2 valid tool but bad args, −1 wrong tool, −0.3 repeat, +0.5 correct stop) and `format_reward`.
* Checkpoints are saved to `<artifacts>/<run>/<algo>/final` and registered as a new agent version with `model="hf:<path>"`;
  `HFLocalProvider` serves them (text-format tool calling), or serve with vLLM and point LiteLLM at it.
* LoRA via `lora_r>0` (merged on save).

## 4. Scaling out (not bundled)

For multi-node PPO/GRPO with vLLM/SGLang rollouts use one of (all permissive licences):

* **verl** – export `grpo_prompts` → parquet; implement the reward as a `verl` custom reward function that calls
  `saphire.training.trl_trainers.replay_reward`. Multi-turn tool calling via verl's agent loop.
* **OpenRLHF** – `--train.agent_func_path` pointing at a function that runs `ToolAgent.run` in a `saphire` environment and returns the verifier reward.
* **agent-lightning / SkyRL / ART** – wrap `ToolAgent` as the rollout function; use `env.verify` for rewards.

The environment abstraction (fresh state per rollout, deterministic tools, final-state verification) is what makes these integrations
straightforward: the reward is always "run the trajectory in the environment and verify".
