import pytest

from saphire.environments import get_environment, list_environments
from saphire.evaluation.gates import GatePolicy, evaluate_gate
from saphire.evaluation.metrics import bootstrap_ci, compare, pass_at_k, pass_hat_k, permutation_test
from saphire.evaluation.runner import evaluate
from saphire.evaluation.suites import build_suite, list_suites
from saphire.sdk import AgentConfig, MockLLM, ToolAgent
from saphire.signals.judges import PairwiseJudge, RubricJudge


def test_environments_registered():
    assert {"support_desk", "data_ops"} <= set(list_environments())
    env = get_environment("support_desk")
    assert len(env.tools) == 30
    assert len([t for t in env.tools.specs() if "core" in t.tags]) == 14


@pytest.mark.parametrize("env_name", ["support_desk", "data_ops"])
def test_perfect_policy_solves_most_tasks(env_name):
    env = get_environment(env_name)
    tasks = env.generate_tasks(21, seed=5)
    res = evaluate(ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM()), tasks, envs={env_name: env})
    assert res.metrics["task_success"] >= 0.75
    assert res.metrics["tool_selection_f1"] >= 0.8
    assert res.metrics["throughput_tasks_per_min"] > 0


def test_verifier_state_based_checks():
    env = get_environment("support_desk")
    task = [t for t in env.generate_tasks(14, seed=2) if t.tags[0] == "cancel_email"][0]
    st = env.reset(task)
    agent = ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM())
    ro = agent.run(task, env, st)
    rewards = {r.name: r for r in env.verify(task, ro, st) if r.step_index is None}
    oid = list(task.expected["order_status"])[0]
    assert rewards["task_success"].metadata["checks"][f"order_{oid}_status"] == (st.data["orders"][oid]["status"] == "cancelled")
    # seed data must be untouched (per-rollout copies)
    assert env.seed_data()["orders"][oid]["status"] != "cancelled"


def test_context_loss_hurts_context_preservation():
    tasks = build_suite("context_preservation", n_per_env=10, seed=3)
    full = evaluate(ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM()), tasks)
    lossy = evaluate(ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM(context_window=2)), tasks)
    assert full.metrics["context_preservation"] >= lossy.metrics["context_preservation"]
    assert lossy.metrics["context_preservation"] < 1.0


def test_pass_hat_k_and_consistency():
    tasks = build_suite("smoke", n_per_env=6, seed=1)
    res = evaluate(ToolAgent(AgentConfig(router_top_k=8), llm=MockLLM(error_rate=0.5, seed=1)), tasks, k=3)
    m = res.metrics
    assert 0 <= m["pass_hat_3"] <= m["task_success"] <= m["pass_at_3"] <= 1
    assert "consistency" in m and res.n_rollouts == len(tasks) * 3
    assert pass_hat_k({"a": [1, 1, 0]}, 2) == pytest.approx(1 / 3)
    assert pass_at_k({"a": [1, 0, 0]}, 2) == pytest.approx(2 / 3)


def test_metrics_helpers():
    lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1])
    assert lo <= 4 / 6 <= hi
    assert permutation_test([0] * 20, [1] * 20) < 0.01
    assert permutation_test([1, 0] * 10, [0, 1] * 10) > 0.5
    c = compare({"per_rollout": [{"task_success": 0.0}] * 10}, {"per_rollout": [{"task_success": 1.0}] * 10}, ["task_success"])
    assert c["task_success"]["delta"] == 1.0


def test_gate_thresholds_and_regression():
    tasks = build_suite("smoke", n_per_env=8, seed=1)
    good = evaluate(ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM()), tasks)
    bad = evaluate(ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM(error_rate=0.9, seed=2)), tasks)
    policy = GatePolicy(min_metrics={"task_success": 0.6}, max_metrics={"error_rate": 0.1})
    assert evaluate_gate(good, policy).passed
    d = evaluate_gate(bad, policy, baseline=good)
    assert not d.passed and any("task_success" in r for r in d.reasons)


def test_suites():
    assert "full" in list_suites()
    tasks = build_suite("long_horizon", n_per_env=4)
    assert all(t.difficulty == "hard" for t in tasks)


def test_judges_offline():
    env = get_environment("support_desk")
    task = env.generate_tasks(3, seed=9)[0]
    agent = ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM())
    st = env.reset(task)
    ro = agent.run(task, env, st)
    ro.rewards = env.verify(task, ro, st)
    r = RubricJudge("mock")(task, ro)
    assert r[0].name == "judge_score" and 0 <= r[0].value <= 1 and "checks" in r[0].metadata
    st2 = env.reset(task)
    ro2 = ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM(error_rate=1.0)).run(task, env, st2)
    ro2.rewards = env.verify(task, ro2, st2)
    assert PairwiseJudge("mock")(task, ro, ro2) >= 0
