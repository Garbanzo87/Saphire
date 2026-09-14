"use client";
import Link from "next/link";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { Api, fetcher, q, type Agent, type EvalRun } from "@/lib/api";
import { ms, pct, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, ErrorBox, IdLink, PageHeader, Panel, Stat } from "@/components/ui";

function AgentDetail() {
  const id = useSearchParams().get("id") || "";
  const { data, error, mutate } = useSWR<Agent>(id ? `/v1/agents/${id}` : null, fetcher);
  const { data: evals, error: evalsErr } = useSWR<EvalRun[]>(id ? q("/v1/evals", { agent_id: id, limit: 50 }) : null, fetcher, { refreshInterval: 5000 });
  const { data: siblings } = useSWR<Agent[]>(data ? q("/v1/agents", { name: data.name }) : null, fetcher);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  if (!id) return <ErrorBox error="Missing ?id=" />;
  const children = (siblings || []).filter((s) => s.parent_id === id);

  async function promote() {
    setBusy(true);
    setErr(null);
    try {
      await Api.promoteAgent(id);
      await mutate();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <DataState data={data} error={error}>
      {(a) => {
        const { system_prompt, ...rest } = a.config || ({} as Agent["config"]);
        return (
          <>
            <PageHeader
              title={`${a.name} / ${a.version}`}
              subtitle={
                <span className="flex flex-wrap items-center gap-2">
                  <Chip status={a.status} /> origin <span className="mono">{a.origin}</span> · <span className="mono">{a.id}</span> · created {ts(a.created_at)}
                </span>
              }
              actions={
                <>
                  <Link href={`/agents?name=${encodeURIComponent(a.name)}`} className="btn">All versions</Link>
                  <button className="btn btn-primary" disabled={a.status === "deployed" || busy} onClick={promote}>
                    {a.status === "deployed" ? "Deployed" : busy ? "Promoting…" : "Promote"}
                  </button>
                </>
              }
            />
            {err ? <div className="mb-3"><ErrorBox error={err} /></div> : null}
            <div className="grid gap-4 xl:grid-cols-3">
              <div className="space-y-4 xl:col-span-2">
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="Model" value={<span className="mono text-[12px]">{a.config?.model}</span>} />
                  <Stat label="Temperature" value={a.config?.temperature ?? "–"} />
                  <Stat label="Max steps" value={a.config?.max_steps ?? "–"} />
                  <Stat label="Router top-k" value={a.config?.tool_router ? a.config.router_top_k ?? "–" : "no router"} />
                </div>
                <Panel title="System prompt">
                  {system_prompt ? (
                    <pre className="mono whitespace-pre-wrap rounded-md bg-[var(--bg)] p-3 text-[12px] leading-5">{system_prompt}</pre>
                  ) : (
                    <div className="text-[var(--muted)]">Empty system prompt.</div>
                  )}
                </Panel>
                <Panel title="Config">
                  <JsonView value={rest} />
                </Panel>
              </div>
              <div className="space-y-4">
                <Panel title="Lineage">
                  <div className="space-y-2 text-[13px]">
                    <div>
                      <span className="label">parent</span>
                      <div><IdLink href={`/agents/detail?id=${a.parent_id}`} id={a.parent_id} n={40} /></div>
                    </div>
                    <div>
                      <span className="label">children</span>
                      {children.length === 0 ? (
                        <div className="text-[var(--muted)]">none</div>
                      ) : (
                        children.map((c) => (
                          <div key={c.id}>
                            <Link href={`/agents/detail?id=${c.id}`} className="link mono text-[12px]">{c.version} · {c.id}</Link>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </Panel>
                <Panel title="Eval runs">
                  <DataState data={evals} error={evalsErr} isEmpty={(d) => d.length === 0} empty="No eval runs for this version.">
                    {(rows) => (
                      <div className="overflow-x-auto">
                        <table className="tbl">
                          <thead>
                            <tr>
                              <th>Run</th>
                              <th>Suite</th>
                              <th>Status</th>
                              <th>Success</th>
                              <th>Tool F1</th>
                              <th>p95</th>
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((e) => (
                              <tr key={e.id}>
                                <td><IdLink href={`/evals/detail?id=${e.id}`} id={e.id} n={18} /></td>
                                <td>{e.suite}</td>
                                <td><Chip status={e.status} /></td>
                                <td className="tabular-nums">{pct(e.metrics?.task_success)}</td>
                                <td className="tabular-nums">{pct(e.metrics?.tool_selection_f1)}</td>
                                <td className="tabular-nums">{ms(e.metrics?.latency_ms_p95)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </DataState>
                </Panel>
              </div>
            </div>
          </>
        );
      }}
    </DataState>
  );
}

export default function Page() {
  return (
    <Suspense>
      <AgentDetail />
    </Suspense>
  );
}
