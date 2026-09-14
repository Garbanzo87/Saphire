import pytest

from saphire.environments import get_environment
from saphire.evaluation.runner import evaluate
from saphire.evaluation.suites import build_suite
from saphire.sdk import AgentConfig, MockLLM, ToolAgent
from saphire.signals.generate import export_jsonl, grpo_prompts, load_jsonl, preference_pairs, router_examples, sft_examples, step_rewards
from saphire.training.learners import train_router, update_exemplars
from saphire.training.online import OnlineLoop
from saphire.training.prompt_opt import mine_rules, optimize_prompt


def _rollouts(n=12, errs=(0.0, 0.6)):
    tasks = build_suite("smoke", n_per_env=n, seed=4)
    envs = {"support_desk": get_environment("support_desk")}
    out = []
    for e in errs:
        evaluate(ToolAgent(AgentConfig(router_top_k=8), llm=MockLLM(error_rate=e, seed=1)), tasks, envs=envs, on_rollout=lambda ro, t, r: out.append(ro))
    return tasks, envs, out


def test_signal_generation(tmp_path):
    tasks, envs, ros = _rollouts()
    tools = {k: v.tools.specs() for k, v in envs.items()}
    rex = router_examples(ros)
    assert rex and any(w < 0 for _, _, w in rex) and any(w > 0 for _, _, w in rex)
    sft = sft_examples(ros, tools)
    assert sft and all("<|assistant|>" in r["prompt"] for r in sft)
    pairs = preference_pairs(ros, tools)
    assert pairs and all(p["chosen"] != p["rejected"] for p in pairs)
    gp = grpo_prompts(ros, {t.id: t for t in tasks}, tools)
    assert gp and "expected_tools" in gp[0]
    assert step_rewards(ros)
    p = export_jsonl(sft, tmp_path / "sft.jsonl")
    assert len(load_jsonl(p)) == len(sft)


def test_router_training_improves_top1(tmp_path):
    tasks, envs, ros = _rollouts()
    r = train_router(ros, envs, tmp_path / "router")
    assert r["examples"] > 0 and r["accuracy"]["support_desk"]["top1"] > 0.5
    ex = update_exemplars(ros, tmp_path / "ex.json")
    assert ex["size"] >= 1


def test_prompt_opt_mines_rules_and_does_not_regress():
    rules = mine_rules([{"confusions": [("track_return", "track_shipment", "track the shipment for me")]}])
    assert rules and "track_shipment" in rules[0]
    tasks = build_suite("smoke", n_per_env=8, seed=2)
    r = optimize_prompt(AgentConfig(model="mock:error=0.5,seed=2", router_top_k=8), tasks, iterations=2, minibatch=8, seed=0)
    assert r["best_score"] >= r["baseline_score"] and r["n_candidates"] >= 1


def test_online_loop_improves(tmp_path):
    train = build_suite("full", n_per_env=28, seed=100)
    ev = build_suite("full", n_per_env=14, seed=200)
    loop = OnlineLoop(AgentConfig(name="t", model="mock:error=0.3,seed=3", router_top_k=8), train, ev, artifacts_dir=tmp_path,
                      batch_size=24, learn_prompt_every=2, seed=1)
    hist = loop.run(4)
    assert hist[0]["iteration"] == 0 and hist[-1]["version"] == "v4"
    assert hist[-1]["eval"]["task_success"] > hist[0]["eval"]["task_success"]
    assert (tmp_path / "t" / "v4" / "agent.json").exists() and (tmp_path / "t" / "history.json").exists()


@pytest.mark.slow
def test_trl_smoke(tmp_path):
    pytest.importorskip("trl")
    from saphire.training import trl_trainers as T

    tasks, envs, ros = _rollouts(n=6)
    tools = {k: v.tools.specs() for k, v in envs.items()}
    r = T.sft(sft_examples(ros, tools)[:4], tmp_path / "sft", max_steps=1)
    assert r["model"].startswith("hf:")
    r = T.dpo(preference_pairs(ros, tools)[:2], tmp_path / "dpo", max_steps=1)
    assert r["algorithm"] == "dpo"
    r = T.grpo(grpo_prompts(ros, {t.id: t for t in tasks}, tools)[:2], tmp_path / "grpo", max_steps=1, num_generations=2, batch_size=2)
    assert "reward_curve" in r
    assert T.replay_reward('<tool_call>{"name": "lookup_order", "arguments": {"order_id": "ORD-2001"}}</tool_call>', "support_desk", "x", [], ["lookup_order"]) == 1.0
    assert T.replay_reward("garbage", "support_desk", "x", [], ["lookup_order"]) < 0
