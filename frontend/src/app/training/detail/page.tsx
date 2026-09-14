"use client";
import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { fetcher, type Metrics, type TrainingRun } from "@/lib/api";
import { delta, isActive, num, pct, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { MetricsGrid } from "@/components/MetricsGrid";
import { MetricLineChart } from "@/components/charts";
import { Chip, DataState, ErrorBox, Expander, IdLink, Logs, PageHeader, Panel, Progress, Stat } from "@/components/ui";

function TrainingView() {
  const id = useSearchParams().get("id") || "";
  const { data, error } = useSWR<TrainingRun>(id ? `/v1/training/${id}` : null, fetcher, {
    refreshInterval: (d) => (isActive(d?.status) || isActive(d?.job?.status) ? 3000 : 0),
  });
  if (!id) return <ErrorBox error="Missing ?id=" />;
  return (
    <DataState data={data} error={error}>
      {(t) => {
        const res = (t.result || {}) as { baseline?: Metrics; final?: Metrics; improvement?: Record<string, number>; iterations?: number };
        const hist = t.history || [];
        const chart = hist.map((h) => ({
          x: `${h.iteration} · ${h.version}`,
          task_success: h.eval?.task_success,
          tool_selection_f1: h.eval?.tool_selection_f1,
          context_preservation: h.eval?.context_preservation,
          collect_success: h.collect_success ?? undefined,
        }));
        const active = isActive(t.status) || isActive(t.job?.status);
        return (
          <>
            <PageHeader
              title={`Training ${t.id}`}
              subtitle={
                <span className="flex flex-wrap items-center gap-2">
                  <Chip status={t.status} /> <span className="mono">{t.algorithm}</span> · input <Link href={`/agents/detail?id=${t.agent_id}`} className="link">{t.agent_id}</Link>
                  {t.output_agent_id && <>· output <Link href={`/agents/detail?id=${t.output_agent_id}`} className="link">{t.output_agent_id}</Link></>} · {ts(t.created_at)}
                </span>
              }
              actions={
                <>
                  <Link href={`/rollouts?training_run_id=${t.id}`} className="btn">Rollouts</Link>
                  <Link href={`/jobs?id=${t.job_id}`} className="btn">Job</Link>
                </>
              }
            />
            {active && (
              <div className="mb-4 flex items-center gap-3 rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-[13px]">
                Running <Progress value={t.job?.progress} /> <span className="text-[var(--muted)]">auto-refreshing</span>
              </div>
            )}
            {t.job?.error && <div className="mb-4"><ErrorBox error={t.job.error} /></div>}
            <div className="space-y-4">
              {res.improvement && (
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Stat label="Baseline task success" value={pct(res.baseline?.task_success)} />
                  <Stat label="Final task success" value={pct(res.final?.task_success)} />
                  {Object.entries(res.improvement).slice(0, 2).map(([k, v]) => (
                    <Stat key={k} label={`Δ ${k}`} value={<span className={v > 0 ? "text-emerald-300" : v < 0 ? "text-rose-300" : ""}>{delta(v)}</span>} />
                  ))}
                </div>
              )}
              {t.algorithm === "online" || hist.length > 0 ? (
                <Panel title="Per-iteration eval">
                  {chart.length === 0 ? (
                    <div className="text-[var(--muted)]">No iterations recorded yet.</div>
                  ) : (
                    <MetricLineChart data={chart} xKey="x" series={[{ key: "task_success" }, { key: "tool_selection_f1" }, { key: "context_preservation" }, { key: "collect_success", label: "collect success (train batch)" }]} height={280} />
                  )}
                </Panel>
              ) : null}
              {hist.length > 0 && (
                <Panel title="Versions created">
                  <div className="overflow-x-auto">
                    <table className="tbl">
                      <thead><tr><th>Iter</th><th>Version</th><th>Agent</th><th>Collect success</th><th>Task success</th><th>Tool F1</th><th>Context</th><th>Step eff.</th><th>Seconds</th></tr></thead>
                      <tbody>
                        {hist.map((h) => (
                          <tr key={h.iteration}>
                            <td className="tabular-nums">{h.iteration}</td>
                            <td className="mono">{h.version}</td>
                            <td><IdLink href={`/agents/detail?id=${h.agent_id}`} id={h.agent_id} n={24} /></td>
                            <td className="tabular-nums">{pct(h.collect_success)}</td>
                            <td className="tabular-nums font-medium">{pct(h.eval?.task_success)}</td>
                            <td className="tabular-nums">{pct(h.eval?.tool_selection_f1)}</td>
                            <td className="tabular-nums">{pct(h.eval?.context_preservation)}</td>
                            <td className="tabular-nums">{pct(h.eval?.step_efficiency)}</td>
                            <td className="tabular-nums">{num(h.seconds, 2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              )}
              {res.final && <Panel title="Final eval metrics"><MetricsGrid m={res.final} /></Panel>}
              <Panel title="Parameters"><JsonView value={t.params} /></Panel>
              <Expander title="Result JSON" defaultOpen={!res.final}>
                <JsonView value={t.result} />
              </Expander>
              <Expander title="Job logs" defaultOpen>
                <Logs logs={t.job?.logs} />
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
      <TrainingView />
    </Suspense>
  );
}
