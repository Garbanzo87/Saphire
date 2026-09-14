# Integrations

## LLM providers
`AgentConfig.model` strings: `mock[:error=..,ctx=..,seed=..,latency=..]` · `hf:<path-or-hub-id>` · anything LiteLLM understands
(`openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-20250514`, `azure/<deployment>`, `bedrock/...`, `ollama/llama3.1`,
`openai/<model>` with `api_base` for vLLM). Keys via the usual env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).

## Tools & environments from your systems
* **SQL** – `SQLConnector(url, tables=[...], writable=[...]).registry()` → `get_<t>`, `search_<t>`, `update_<t>`, `insert_<t>`, `run_readonly_sql`.
* **REST / OpenAPI** – `OpenAPIConnector(spec_url_or_path, base_url, headers).registry()` → one tool per operation.
* **MCP** – `registry_from_mcp(command, args, env)` (stdio servers; `pip install saphire[mcp]`).
* **Exports** – `FileDataConnector({"orders": "orders.csv"}, writable=("orders",))` → safe, per-rollout copies of the data.
* Combine registries with `reg.merge(other)`; wrap into `DataEnvironment(name, registry, seed_data, tasks, verifier)`.

## Observability
* `tracing.init(service_name, host, api_key, project)` exports OTel spans to `POST /v1/traces/ingest`.
* OpenInference/OTel instrumentors (`openinference-instrumentation-openai`, `-anthropic`, `-langchain`, ...) can be attached to the
  provider returned by `tracing.init()`; their spans land in the same project.
* Human / product feedback: `POST /v1/scores` (by `trace_id` or `rollout_id`).

## Bring-your-own agent
Record steps with `RolloutRecorder`, verify with your environment (or attach scores), then `SaphireClient.log_rollout(...)`.
See `examples/byo_agent.py`. Training jobs (`router`, `exemplars`, `signals`, `sft`, `dpo`, `grpo`) work on any stored rollouts
for the agent name.
