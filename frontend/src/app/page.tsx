"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { fetcher, q, type Overview, type TimeseriesPoint } from "@/lib/api";
import { ago, isNum, pct, ts } from "@/lib/format";
import { Chip, DataState, KPI, PageHeader, Panel, Progress } from "@/components/ui";
import { MetricLineChart } from "@/components/charts";

const METRICS = ["task_success", "tool_selection_f1", "context_preservation", "step_efficiency"];

function AgentChart({ name, metric }: { name: string; metric: string }) {
  const { data, error } = useSWR<TimeseriesPoint[]>(q("/v1/metrics/timeseries", { agent_name: name, name: metric, limit: 200 }), fetcher, { refreshInterval: 5000 });
  const rows = useMemo(
    () =>
      (data || [])
        .slice()
        .sort((a, b) => a.ts - b.ts)
        .map((p, i) => ({
          x: `${p.version} · ${new Date(p.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`,
          i,
          value: p.value,
          source: p.tags?.eval_run_id ? `eval ${String(p.tags.suite ?? "")}` : p.tags?.training_run_id ? `train iter ${String(p.tags.iteration ?? "")}` : "",
        })),
    [data],
  );
  return (
    <DataState data={data} error={error} isEmpty={() => rows.length === 0} empty={`No ${metric} points for ${name}.`}>
      {() => <MetricLineChart data={rows} xKey="x" series={[{ key: "value", label: metric }]} height={220} />}
    </DataState>
  );
}

export default function OverviewPage() {
  const { data, error } = useSWR<Overview>("/v1/metrics/overview", fetcher, { refreshInterval: 5000 });
  const [metric, setMetric] = useState(METRICS[0]);
  return (
    <>
      <PageHeader title="Overview" subtitle={data ? `Project ${data.project.name} · ${data.project.id}` : undefined} />
      <DataState data={data} error={error}>
        {(o) => {
          const agents = Object.entries(o.agents || {});
          return (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
                <KPI label="Agents" value={o.n_agents} hint={`${agents.length} name${agents.length === 1 ? "" : "s"}`} />
                <KPI label="Traces" value={o.n_traces.toLocaleString()} />
                <KPI label="Rollouts" value={o.n_rollouts.toLocaleString()} />
                <KPI label="Rollout success" value={pct(o.rollout_success_rate)} />
                <KPI label="Eval runs" value={o.n_eval_runs} />
                <KPI label="Training runs" value={o.n_training_runs} />
              </div>

              {agents.length === 0 ? (
                <Panel title="Agents">
                  <div className="text-[var(--muted)]">No agents registered in this project.</div>
                </Panel>
              ) : (
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {agents.map(([name, a]) => {
                    const first = a.first_task_success;
                    const latest = a.latest_task_success;
                    const d = isNum(first) && isNum(latest) ? latest - first : null;
                    return (
                      <div key={name} className="panel p-4">
                        <div className="flex items-start justify-between">
                          <div>
                            <Link href={`/agents?name=${encodeURIComponent(name)}`} className="text-[15px] font-semibold hover:underline">
                              {name}
                            </Link>
                            <div className="text-[12px] text-[var(--muted)]">
                              {a.versions} version{a.versions === 1 ? "" : "s"}
                            </div>
                          </div>
                          {a.deployed_version ? (
                            <Link href={`/agents/detail?id=${a.deployed_agent_id}`}>
                              <Chip status="deployed">deployed {a.deployed_version}</Chip>
                            </Link>
                          ) : (
                            <Chip status="candidate">not deployed</Chip>
                          )}
                        </div>
                        <div className="mt-3 grid grid-cols-3 gap-2">
                          <div>
                            <div className="label">first</div>
                            <div className="text-[15px] font-semibold tabular-nums">{pct(first)}</div>
                          </div>
                          <div>
                            <div className="label">latest</div>
                            <div className="text-[15px] font-semibold tabular-nums">{pct(latest)}</div>
                          </div>
                          <div>
                            <div className="label">task success Δ</div>
                            <div className={`text-[15px] font-semibold tabular-nums ${d === null ? "" : d > 0 ? "text-emerald-300" : d < 0 ? "text-rose-300" : "text-[var(--muted)]"}`}>
                              {d === null ? "–" : `${d > 0 ? "▲" : d < 0 ? "▼" : "•"} ${(Math.abs(d) * 100).toFixed(1)} pp`}
                            </div>
                          </div>
                        </div>
                        {a.latest_eval && (
                          <div className="mt-3 text-[12px] text-[var(--muted)]">
                            latest eval{" "}
                            <Link href={`/evals/detail?id=${a.latest_eval.id}`} className="link mono">
                              {a.latest_eval.id}
                            </Link>{" "}
                            · {a.latest_eval.suite} · {pct(a.latest_eval.metrics?.task_success)} · {ago(a.latest_eval.created_at)}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {agents.length > 0 && (
                <Panel
                  title="Metric over time"
                  actions={
                    <select className="input" value={metric} onChange={(e) => setMetric(e.target.value)}>
                      {METRICS.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  }
                >
                  <div className="grid gap-4 xl:grid-cols-2">
                    {agents.map(([name]) => (
                      <div key={name}>
                        <div className="mb-1 text-[12px] font-medium">{name}</div>
                        <AgentChart name={name} metric={metric} />
                      </div>
                    ))}
                  </div>
                </Panel>
              )}

              <Panel title="Recent jobs" actions={<Link href="/jobs" className="link text-[12px]">all jobs →</Link>}>
                {o.recent_jobs?.length ? (
                  <div className="overflow-x-auto">
                    <table className="tbl">
                      <thead>
                        <tr>
                          <th>Job</th>
                          <th>Type</th>
                          <th>Status</th>
                          <th>Progress</th>
                          <th>Target</th>
                          <th>Created</th>
                        </tr>
                      </thead>
                      <tbody>
                        {o.recent_jobs.map((j) => {
                          const p = j.params as Record<string, string>;
                          const target = p.eval_run_id ? (
                            <Link href={`/evals/detail?id=${p.eval_run_id}`} className="link mono text-[12px]">{p.eval_run_id}</Link>
                          ) : p.training_run_id ? (
                            <Link href={`/training/detail?id=${p.training_run_id}`} className="link mono text-[12px]">{p.training_run_id}</Link>
                          ) : (
                            "–"
                          );
                          return (
                            <tr key={j.id}>
                              <td className="mono text-[12px]"><Link href={`/jobs?id=${j.id}`} className="link">{j.id}</Link></td>
                              <td>{j.type}</td>
                              <td><Chip status={j.status} /></td>
                              <td><Progress value={j.progress} /></td>
                              <td>{target}</td>
                              <td className="whitespace-nowrap text-[var(--muted)]">{ts(j.created_at)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-[var(--muted)]">No jobs yet.</div>
                )}
              </Panel>
            </div>
          );
        }}
      </DataState>
    </>
  );
}
