"""SQL connector: expose database tables as read (and optionally write) tools."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from sqlalchemy import MetaData, Table, create_engine, insert, select, text, update

from ..sdk.tools import ToolRegistry
from ..sdk.types import ToolSpec


class SQLConnector:
    """
        conn = SQLConnector("postgresql://.../crm", tables=["customers", "orders"], writable=["orders"])
        registry = conn.registry()   # get_customers_by_<pk>, search_customers, list_orders, update_orders ...
    """

    def __init__(self, url: str, tables: Optional[Iterable[str]] = None, writable: Iterable[str] = (), max_rows: int = 20,
                 tag: str = "sql"):
        self.engine = create_engine(url)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine, only=list(tables) if tables else None)
        self.writable = set(writable)
        self.max_rows = max_rows
        self.tag = tag

    def _rows(self, result) -> list[dict[str, Any]]:
        return [dict(r._mapping) for r in result]

    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        for name, table in self.meta.tables.items():
            self._add_table_tools(reg, name, table)
        reg.add_spec(ToolSpec(name="run_readonly_sql", description="Run a read-only SELECT query against the database.", source="sql",
                              tags=[self.tag], parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
                     self._readonly_sql)
        return reg

    def _readonly_sql(self, query: str) -> list[dict[str, Any]]:
        q = query.strip().rstrip(";")
        if not q.lower().startswith("select"):
            raise ValueError("only SELECT statements are allowed")
        with self.engine.connect() as c:
            return self._rows(c.execute(text(q)).fetchmany(self.max_rows))

    def _add_table_tools(self, reg: ToolRegistry, name: str, table: Table) -> None:
        pk = [c.name for c in table.primary_key.columns] or [table.columns.keys()[0]]
        cols = {c.name: _jtype(c.type) for c in table.columns}

        def get(**kw):
            with self.engine.connect() as c:
                stmt = select(table).where(*[table.c[k] == v for k, v in kw.items() if k in table.c])
                rows = self._rows(c.execute(stmt).fetchmany(self.max_rows))
            if not rows:
                raise KeyError(f"no {name} row matching {kw}")
            return rows[0] if len(rows) == 1 else rows

        reg.add_spec(ToolSpec(name=f"get_{name}", description=f"Get a {name} row by {'/'.join(pk)}.", source="sql", tags=[self.tag],
                              parameters={"type": "object", "properties": {k: {"type": cols[k]} for k in pk}, "required": pk}), get)

        def search(**kw):
            with self.engine.connect() as c:
                conds = []
                for k, v in kw.items():
                    if k in table.c and v not in (None, ""):
                        conds.append(table.c[k].like(f"%{v}%") if cols[k] == "string" else table.c[k] == v)
                stmt = select(table).where(*conds).limit(self.max_rows)
                return self._rows(c.execute(stmt))

        reg.add_spec(ToolSpec(name=f"search_{name}", description=f"Search {name} rows by any column value (partial match for text).",
                              source="sql", tags=[self.tag],
                              parameters={"type": "object", "properties": {k: {"type": t} for k, t in cols.items()}, "required": []}), search)
        if name in self.writable:
            def upd(**kw):
                key = {k: kw.pop(k) for k in pk}
                with self.engine.begin() as c:
                    res = c.execute(update(table).where(*[table.c[k] == v for k, v in key.items()]).values(**kw))
                return {"updated": res.rowcount, **key, **kw}

            reg.add_spec(ToolSpec(name=f"update_{name}", description=f"Update columns of a {name} row identified by {'/'.join(pk)}.",
                                  source="sql", tags=[self.tag, "write"],
                                  parameters={"type": "object", "properties": {k: {"type": t} for k, t in cols.items()}, "required": pk}), upd)

            def ins(**kw):
                with self.engine.begin() as c:
                    c.execute(insert(table).values(**kw))
                return {"inserted": 1, **kw}

            reg.add_spec(ToolSpec(name=f"insert_{name}", description=f"Insert a new {name} row.", source="sql", tags=[self.tag, "write"],
                                  parameters={"type": "object", "properties": {k: {"type": t} for k, t in cols.items()},
                                              "required": [k for k in cols if k not in pk]}), ins)


def _jtype(sqltype) -> str:
    s = str(sqltype).lower()
    if any(x in s for x in ("int",)):
        return "integer"
    if any(x in s for x in ("float", "numeric", "decimal", "real", "double")):
        return "number"
    if "bool" in s:
        return "boolean"
    return "string"
