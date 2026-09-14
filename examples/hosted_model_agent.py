"""Evaluate a hosted model (any LiteLLM provider) with an LLM judge on the built-in suites.

    export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY, or run Ollama locally
    python examples/hosted_model_agent.py openai/gpt-4o-mini
"""
import sys

from saphire.evaluation.runner import evaluate
from saphire.evaluation.suites import build_suite
from saphire.sdk import AgentConfig, ToolAgent
from saphire.signals.judges import RubricJudge

model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
agent = ToolAgent(AgentConfig(name="hosted", model=model, router_top_k=12, temperature=0.0))
tasks = build_suite("tool_selection", n_per_env=6, seed=0)
res = evaluate(agent, tasks, k=1, judge=RubricJudge(model), concurrency=4)
print({k: round(v, 3) for k, v in res.metrics.items() if isinstance(v, float)})
