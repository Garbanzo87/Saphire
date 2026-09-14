"use client";
import Link from "next/link";
import { Fragment, useState } from "react";
import useSWR from "swr";
import { Api, fetcher, type Agent, type Deployment } from "@/lib/api";
import { delta, ms, pct, pval, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, ErrorBox, Expander, IdLink, PageHeader, Panel } from "@/components/ui";

function Checks({ d }: { d: Deployment }) {
  const checks = Object.entries(d.decision?.checks || {});
  return (
    <div className="space-y-3">
      {d.decision?.reasons?.length > 0 && (
        <ul className="list-disc pl-5 text-[13px] text-rose-300">
          {d.decision.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}
      <table className="tbl">
        <thead><tr><th>Check</th><th>Value</th><th>Threshold</th><th>Δ vs baseline</th><th>p</th><th>OK</th></tr></thead>
        <tbody>
          {checks.map(([k, c]) => (
            <tr key={k}>
              <td className="mono text-[12px]">{k}</td>
              <td className="tabular-nums">{c.value !== undefined ? (k.includes("latency") ? ms(c.value) : pct(c.value)) : "–"}</td>
              <td className="tabular-nums">{c.threshold !== undefined ? (k.includes("latency") ? ms(c.threshold) : pct(c.threshold)) : "–"}</td>
              <td className={`tabular-nums ${c.delta !== undefined ? (c.delta > 0 ? "text-emerald-300" : c.delta < 0 ? "text-rose-300" : "") : ""}`}>{c.delta !== undefined ? delta(c.delta) : "–"}</td>
              <td className="tabular-nums">{c.p_value !== undefined ? pval(c.p_value) : "–"}</td>
              <td><Chip status={c.ok}>{c.ok ? "ok" : "fail"}</Chip></td>
            </tr>
          ))}
        </tbody>
      </table>
      <Expander title="Policy">
        <JsonView value={d.policy} />
      </Expander>
    </div>
  );
}

export default function DeploymentsPage() {
  const { data, error, mutate } = useSWR<Deployment[]>("/v1/deployments", fetcher, { refreshInterval: 5000 });
  const { data: current, error: curErr, mutate: mutateCur } = useSWR<Agent[]>("/v1/deployments/current", fetcher, { refreshInterval: 5000 });
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [open, setOpen] = useState<string | null>(null);

  async function promote(id: string) {
    setBusy(id);
    setErr(null);
    try {
      await Api.promoteDeployment(id);
      await Promise.all([mutate(), mutateCur()]);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHeader title="Deployments" subtitle="Gate decisions and currently deployed agent versions" />
      {err ? <div className="mb-3"><ErrorBox error={err} /></div> : null}
      <div className="space-y-4">
        <Panel title="Currently deployed">
          <DataState data={current} error={curErr} isEmpty={(d) => d.length === 0} empty="No agent is deployed.">
            {(agents) => (
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                {agents.map((a) => (
                  <Link key={a.id} href={`/agents/detail?id=${a.id}`} className="flex items-center justify-between rounded-md border border-[var(--border)] bg-[var(--panel-2)] px-3 py-2 hover:border-[var(--accent)]">
                    <div>
                      <div className="font-medium">{a.name} <span className="mono text-[var(--muted)]">{a.version}</span></div>
                      <div className="mono text-[11px] text-[var(--muted)]">{a.id} · {a.origin}</div>
                    </div>
                    <Chip status="deployed" />
                  </Link>
                ))}
              </div>
            )}
          </DataState>
        </Panel>
        <Panel title="Gate decisions">
          <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No gate decisions yet — run an evaluation with the gate enabled.">
            {(rows) => (
              <div className="overflow-x-auto">
                <table className="tbl">
                  <thead>
                    <tr><th>Decision</th><th>Agent</th><th>Candidate</th><th>Baseline</th><th>Result</th><th>Task success</th><th>Tool F1</th><th>Promoted</th><th>Active</th><th>Eval</th><th>Created</th><th></th></tr>
                  </thead>
                  <tbody>
                    {rows.map((d) => {
                      const ts_ = d.decision?.checks?.["min:task_success"];
                      const f1 = d.decision?.checks?.["min:tool_selection_f1"];
                      const expanded = open === d.id;
                      return (
                        <Fragment key={d.id}>
                          <tr className="cursor-pointer" onClick={() => setOpen(expanded ? null : d.id)}>
                            <td className="mono text-[12px]"><span className="mr-1 text-[var(--muted)]">{expanded ? "▾" : "▸"}</span>{d.id}</td>
                            <td className="font-medium">{d.agent_name}</td>
                            <td><IdLink href={`/agents/detail?id=${d.agent_id}`} id={d.agent_version || d.agent_id} n={24} /></td>
                            <td>{d.baseline_agent_id ? <IdLink href={`/agents/detail?id=${d.baseline_agent_id}`} id={d.baseline_version || d.baseline_agent_id} n={24} /> : <span className="text-[var(--muted)]">none</span>}</td>
                            <td><Chip status={d.decision?.passed}>{d.decision?.passed ? "passed" : "failed"}</Chip></td>
                            <td className="tabular-nums">{pct(ts_?.value)}</td>
                            <td className="tabular-nums">{pct(f1?.value)}</td>
                            <td><Chip status={d.promoted ? "deployed" : "unset"}>{d.promoted ? "yes" : "no"}</Chip></td>
                            <td><Chip status={d.active ? "succeeded" : "retired"}>{d.active ? "active" : "inactive"}</Chip></td>
                            <td><IdLink href={`/evals/detail?id=${d.eval_run_id}`} id={d.eval_run_id} n={18} /></td>
                            <td className="whitespace-nowrap text-[var(--muted)]">{ts(d.created_at)}</td>
                            <td onClick={(e) => e.stopPropagation()}>
                              <button className="btn btn-primary" disabled={d.promoted || !d.decision?.passed || busy === d.id} onClick={() => promote(d.id)}>
                                {d.promoted ? "Promoted" : busy === d.id ? "Promoting…" : "Promote"}
                              </button>
                            </td>
                          </tr>
                          {expanded && (
                            <tr>
                              <td colSpan={12} className="bg-[var(--bg)]"><Checks d={d} /></td>
                            </tr>
                          )}
                        </Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </DataState>
        </Panel>
      </div>
    </>
  );
}
