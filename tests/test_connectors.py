import json

import httpx
from sqlalchemy import create_engine, text

from saphire.connectors import DataEnvironment, FileDataConnector, OpenAPIConnector, SQLConnector
from saphire.evaluation.runner import evaluate
from saphire.sdk import AgentConfig, MockLLM, TaskSpec, ToolAgent, ToolCall


def test_sql_connector(tmp_path):
    url = f"sqlite:///{tmp_path}/crm.db"
    eng = create_engine(url)
    with eng.begin() as c:
        c.execute(text("CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT, tier TEXT)"))
        c.execute(text("INSERT INTO customers VALUES ('CUS-1', 'Ada', 'gold'), ('CUS-2', 'Bob', 'standard')"))
    reg = SQLConnector(url, writable=["customers"]).registry()
    assert {"get_customers", "search_customers", "update_customers", "insert_customers", "run_readonly_sql"} <= set(reg.names())
    assert reg.call(ToolCall(name="get_customers", arguments={"customer_id": "CUS-1"})).output["name"] == "Ada"
    assert len(reg.call(ToolCall(name="search_customers", arguments={"name": "o"})).output) == 1
    assert reg.call(ToolCall(name="update_customers", arguments={"customer_id": "CUS-2", "tier": "gold"})).output["updated"] == 1
    assert reg.call(ToolCall(name="run_readonly_sql", arguments={"query": "DROP TABLE customers"})).error


def test_openapi_connector():
    spec = {"openapi": "3.0.0", "servers": [{"url": "http://api.test"}],
            "paths": {"/orders/{id}": {"get": {"operationId": "getOrder", "summary": "Get order", "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}]}},
                      "/refunds": {"post": {"operationId": "createRefund", "summary": "Create refund",
                                            "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Refund"}}}}}}},
            "components": {"schemas": {"Refund": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}}, "required": ["order_id", "amount"]}}}}
    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path, request.content))
        return httpx.Response(200, json={"ok": True, "path": request.url.path})

    client = httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(handler))
    reg = OpenAPIConnector(spec, client=client).registry()
    assert reg.get("createRefund").spec.parameters["required"] == ["order_id", "amount"]
    out = reg.call(ToolCall(name="getOrder", arguments={"id": "ORD-1"})).output
    assert out["path"] == "/orders/ORD-1"
    reg.call(ToolCall(name="createRefund", arguments={"order_id": "ORD-1", "amount": 5}))
    assert json.loads(calls[-1][2]) == {"order_id": "ORD-1", "amount": 5}


def test_file_connector_and_data_environment(tmp_path):
    p = tmp_path / "orders.csv"
    p.write_text("order_id,status,customer\nORD-1,processing,Ada\nORD-2,shipped,Bob\n")
    conn = FileDataConnector({"orders": p}, keys={"orders": "order_id"}, writable=("orders",))
    env = DataEnvironment("crm_export", conn.registry(), conn.seed_data(), tasks=[
        TaskSpec(env_name="crm_export", instruction="Get orders ORD-1, then update orders ORD-1 to status cancelled",
                 expected={"tools": ["get_orders", "update_orders"], "state_checks": [{"path": "tables.orders.0.status", "equals": "cancelled"}]}, tags=["update"]),
    ])
    res = evaluate(ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM()), env.generate_tasks(), envs={"crm_export": env})
    assert res.metrics["task_success"] == 1.0
    assert conn.tables["orders"][0]["status"] == "processing"  # source data never mutated
