"""File connector: CSV / JSON exports become in-memory tables with get/search/list/update tools.

Ideal for building a *simulated* environment from a data export so agents can be trained
and evaluated safely without touching production systems (updates apply to a per-rollout
copy of the data — see `DataEnvironment`)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from ..sdk.tools import ToolRegistry
from ..sdk.types import ToolSpec


class FileDataConnector:
    def __init__(self, tables: dict[str, str | Path | list[dict[str, Any]]], keys: Optional[dict[str, str]] = None,
                 writable: tuple[str, ...] = (), max_rows: int = 20):
        self.tables: dict[str, list[dict[str, Any]]] = {n: self._load(src) for n, src in tables.items()}
        self.keys = keys or {n: (list(rows[0].keys())[0] if rows else "id") for n, rows in self.tables.items()}
        self.writable = set(writable)
        self.max_rows = max_rows

    @staticmethod
    def _load(src) -> list[dict[str, Any]]:
        if isinstance(src, list):
            return src
        p = Path(src)
        if p.suffix.lower() == ".csv":
            with p.open() as f:
                return [dict(r) for r in csv.DictReader(f)]
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else list(data.values())

    def seed_data(self) -> dict[str, Any]:
        return {"tables": {n: [dict(r) for r in rows] for n, rows in self.tables.items()}, "log": []}

    def registry(self) -> ToolRegistry:
        """Tools take `state` (an EnvState whose .data == seed_data()) so writes are per-rollout."""
        reg = ToolRegistry()
        for name, rows in self.tables.items():
            key = self.keys[name]
            cols = list(rows[0].keys()) if rows else [key]
            self._add(reg, name, key, cols)
        return reg

    def _add(self, reg: ToolRegistry, name: str, key: str, cols: list[str]) -> None:
        max_rows = self.max_rows

        def get(state, **kw):
            for r in state.data["tables"][name]:
                if str(r.get(key)) == str(kw.get(key)):
                    return r
            raise KeyError(f"no {name} with {key}={kw.get(key)}")

        def search(state, **kw):
            out = []
            for r in state.data["tables"][name]:
                if all(str(v).lower() in str(r.get(k, "")).lower() for k, v in kw.items() if v not in (None, "")):
                    out.append(r)
                if len(out) >= max_rows:
                    break
            return out

        def list_(state):
            return state.data["tables"][name][:max_rows]

        reg.add_spec(ToolSpec(name=f"get_{name}", description=f"Get a {name} record by {key}.", source="file",
                              parameters={"type": "object", "properties": {key: {"type": "string"}}, "required": [key]}), get)
        reg.get(f"get_{name}").takes_state = True
        reg.add_spec(ToolSpec(name=f"search_{name}", description=f"Search {name} records by any field (partial match).", source="file",
                              parameters={"type": "object", "properties": {c: {"type": "string"} for c in cols}, "required": []}), search)
        reg.get(f"search_{name}").takes_state = True
        reg.add_spec(ToolSpec(name=f"list_{name}", description=f"List {name} records.", source="file",
                              parameters={"type": "object", "properties": {}, "required": []}), list_)
        reg.get(f"list_{name}").takes_state = True
        if name in self.writable:
            def upd(state, **kw):
                r = get(state, **{key: kw[key]})
                r.update({k: v for k, v in kw.items() if k != key})
                state.data["log"].append({"table": name, key: kw[key], "update": {k: v for k, v in kw.items() if k != key}})
                return r

            reg.add_spec(ToolSpec(name=f"update_{name}", description=f"Update fields of a {name} record identified by {key}.", source="file",
                                  tags=["write"], parameters={"type": "object", "properties": {c: {"type": "string"} for c in cols}, "required": [key]}), upd)
            reg.get(f"update_{name}").takes_state = True
