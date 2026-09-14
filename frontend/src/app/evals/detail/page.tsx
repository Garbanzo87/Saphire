"use client";
import Link from "next/link";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { fetcher, q, type Comparison, type EvalDetail, type EvalRun, type Metrics } from "@/lib/api";
import { delta, isActive, ms, pct, pval, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { MetricsGrid } from "@/components/MetricsGrid";
import { MetricBarChart } from "@/components/charts";
import { Chip, DataState, ErrorBox, Expander, IdLink, Logs, PageHeader, Panel, Progress } from "@/components/ui";

type Gate = { passed: boolean; reasons: string[]; checks: Record<string, { value?: number; threshold?: number; delta?: number; p_value?: number; ok: boolean }> };

function Breakdown({ title, data }: { title: string; data?: Record<string, Metrics> }) {
  const rows = Object.entries(data || {}).map(([k, m]) => ({ name: `${k} (n=${m.n ?? "?"})`, task_success: m.task_success, tool_selection_f1: m.tool_selection_f1, context_preservation: m.context_preservation }));
  return (
    <Panel title={title}>
      {rows.length === 0 ? <div className="text-[var(--muted)]">No breakdown.</div> : <MetricBarChart data={rows} xKey="name" series={[{ key: "task_success" }, { key: "tool_selection_f1" }, { key: "context_preservation" }]} height={200} />}
    </Panel>
  );
}

function GateView({ gate }: { gate: Gate }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Chip status={gate.passed}>{gate.passed ? "gate passed" : "gate failed"}</Chip>
        {gate.reasons?.length > 0 && <span className="text-[12px] text-rose-300">{gate.reasons.join("; ")}</span>}
      </div>
      <table className="tbl">
        <thead><tr><th>Check</th><th>Value</th><th>Threshold</th><th>Δ</th><th>p</th><th>OK</th></tr></thead>
        <tbody>
          {Object.entries(gate.checks || {}).map(([k, c]) => (
            <tr key={k}>
              <td className="mono text-[12px]">{k}</td>
              <td className="tabular-nums">{c.value !== undefined ? (k.includes("latency") ? ms(c.value) : pct(c.value)) : "–"}</td>
              <td className="tabular-nums">{c.threshold !== undefined ? (k.includes("latency") ? ms(c.threshold) : pct(c.threshold)) : "–"}</td>
              <td className="tabular-nums">{c.delta !== undefined ? delta(c.delta) : "–"}</td>
              <td className="tabular-nums">{c.p_value !== undefined ? pval(c.p_value) : "–"}</td>
              <td><Chip status={c.ok}>{c.ok ? "ok" : "fail"}</Chip></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Compare({ id, others }: { id: string; others: EvalRun[] }) {
  const [other, setOther] = useState("");
  const { data, error } = useSWR<Comparison>(other ? `/v1/evals/${id}/compare/${other}` : null, fetcher);
  return (
    <Panel
      title="Compare with…"
      actions={
        <select className="input" value={other} onChange={(e) => setOther(e.target.value)}>
          <option value="">select an eval run</option>
          {others.map((o) => <option key={o.id} value={o.id}>{o.agent_name}/{o.agent_version} · {o.suite} · {pct(o.metrics?.task_success)} · {o.id}</option>)}
        </select>
      }
    >
      {!other ? (
        <div className="text-[var(--muted)]">Pick another run to see metric deltas (this run is the baseline, the selected run is the candidate).</div>
      ) : (
        <DataState data={data} error={error}>
          {(c) => (
            <table className="tbl">
              <thead><tr><th>Metric</th><th>This run</th><th>Other</th><th>Δ</th><th>p-value</th><th>Other 95% CI</th></tr></thead>
              <tbody>
                {Object.entries(c.comparison).map(([k, v]) => (
                  <tr key={k}>
                    <td className="mono text-[12px]">{k}</td>
                    <td className="tabular-nums">{pct(v.baseline)}</td>
                    <td className="tabular-nums">{pct(v.candidate)}</td>
                    <td className={`tabular-nums ${v.delta > 0 ? "text-emerald-300" : v.delta < 0 ? "text-rose-300" : ""}`}>{delta(v.delta)}</td>
                    <td className={`tabular-nums ${v.p_value < 0.05 ? "font-semibold" : "text-[var(--muted)]"}`}>{pval(v.p_value)}</td>
                    <td className="tabular-nums text-[var(--muted)]">[{pct(v.ci_candidate?.[0])}, {pct(v.ci_candidate?.[1])}]</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </DataState>
      )}
    </Panel>
  );
}

function EvalView() {
  const id = useSearchParams().get("id") || "";
  const { data, error } = useSWR<EvalDetail>(id ? `/v1/evals/${id}` : null, fetcher, {
    refreshInterval: (d) => (isActive(d?.status) || isActive(d?.job?.status) ? 3000 : 0),
  });
  const { data: all } = useSWR<EvalRun[]>(q("/v1/evals", { limit: 100 }), fetcher);
  if (!id) return <ErrorBox error="Missing ?id=" />;
  return (
    <DataState data={data} error={error}>
      {(e) => {
        const gate = (e.job?.result as { gate?: Gate } | null)?.gate;
        const deploymentId = (e.job?.result as { deployment_id?: string } | null)?.deployment_id;
        const agentLabel = e.agent ? `${e.agent.name}/${e.agent.version}` : e.agent_id;
        return (
          <>
            <PageHeader
              title={`Eval ${e.id}`}
              subtitle={
                <span className="flex flex-wrap items-center gap-2">
                  <Chip status={e.status} /> <Link href={`/agents/detail?id=${e.agent_id}`} className="link">{agentLabel}</Link> · suite {e.suite} · k={e.k} · judge {e.judge_model || "none"} · dataset <span className="mono">{e.dataset_id}</span> · {ts(e.created_at)}
                </span>
              }
              actions={
                <>
                  <Link href={`/rollouts?eval_run_id=${e.id}`} className="btn">Rollouts</Link>
                  <Link href={`/jobs?id=${e.job_id}`} className="btn">Job</Link>
                </>
              }
            />
            {isActive(e.status) && (
              <div className="mb-4 flex items-center gap-3 rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-[13px]">
                Running <Progress value={e.job?.progress} /> <span className="text-[var(--muted)]">auto-refreshing</span>
              </div>
            )}
            {e.job?.error && <div className="mb-4"><ErrorBox error={e.job.error} /></div>}
            <div className="space-y-4">
              <Panel title="Metrics"><MetricsGrid m={e.metrics} /></Panel>
              {gate && (
                <Panel title="Gate decision" actions={deploymentId ? <Link href="/deployments" className="link text-[12px]">deployment {deploymentId} →</Link> : null}>
                  <GateView gate={gate} />
                </Panel>
              )}
              <div className="grid gap-4 xl:grid-cols-3">
                <Breakdown title="By family" data={e.by_family} />
                <Breakdown title="By environment" data={e.by_env} />
                <Breakdown title="By difficulty" data={e.by_difficulty} />
              </div>
              <Compare id={e.id} others={(all || []).filter((o) => o.id !== e.id && o.status === "succeeded")} />
              <Panel title={`Rollouts · ${e.rollouts?.length ?? 0}`}>
                {!e.rollouts?.length ? (
                  <div className="text-[var(--muted)]">No rollouts recorded yet.</div>
                ) : (
                  <div className="max-h-[36rem] overflow-auto">
                    <table className="tbl">
                      <thead>
                        <tr><th>Rollout</th><th>Env</th><th>Family</th><th>Difficulty</th><th>Trial</th><th>Success</th><th>Tool F1</th><th>Context</th><th>Steps</th><th>Latency</th><th>Rationale</th><th>Trace</th></tr>
                      </thead>
                      <tbody>
                        {e.rollouts.map((r) => (
                          <tr key={r.rollout_id}>
                            <td><IdLink href={`/rollouts/detail?id=${r.rollout_id}`} id={r.rollout_id} n={20} /></td>
                            <td>{r.env_name}</td>
                            <td>{r.family}</td>
                            <td>{r.difficulty}</td>
                            <td className="tabular-nums">{r.trial}</td>
                            <td className={`tabular-nums ${r.task_success >= 1 ? "text-emerald-300" : "text-rose-300"}`}>{pct(r.task_success)}</td>
                            <td className="tabular-nums">{pct(r.tool_selection_f1)}</td>
                            <td className="tabular-nums">{pct(r.context_preservation)}</td>
                            <td className="tabular-nums">{r.n_steps}</td>
                            <td className="tabular-nums">{ms(r.latency_ms)}</td>
                            <td className="max-w-[20rem] truncate text-[11px] text-[var(--muted)]" title={r.rationale}>{r.rationale}</td>
                            <td><IdLink href={`/traces/detail?id=${r.trace_id}`} id={r.trace_id} n={8} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Panel>
              <Expander title="Job logs">
                <Logs logs={e.job?.logs} />
              </Expander>
              <Expander title="Job result JSON">
                <JsonView value={e.job?.result ?? null} />
              </Expander>
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
      <EvalView />
    </Suspense>
  );
}
