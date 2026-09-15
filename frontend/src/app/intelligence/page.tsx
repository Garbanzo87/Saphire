"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { Api, ApiError, fetcher, q, type Agent, type ApplyResult, type Dataset, type FailureCluster, type IntelligenceReport, type IntelligenceReportSummary, type MineResult, type Recommendation } from "@/lib/api";
import { delta, num, pct, pval, ts } from "@/lib/format";
import { Chip, DataState, ErrorBox, Field, IdLink, KPI, PageHeader, Panel } from "@/components/ui";

const severityChip: Record<string, string> = { high: "failed", medium: "4xx", low: "candidate" };

function RunForm({ onDone }: { onDone: (r: IntelligenceReport) => void }) {
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const names = Array.from(new Set((agents || []).map((a) => a.name)));
  const [form, setForm] = useState({ agent_name: "", recent_hours: 24, baseline_hours: 168, production_only: false });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await Api.runIntelligence({ agent_name: form.agent_name || null, recent_hours: form.recent_hours, baseline_hours: form.baseline_hours, production_only: form.production_only });
      onDone(r);
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form onSubmit={submit} className="grid gap-3 md:grid-cols-5">
      <Field label="Agent (optional)">
        <select className="input" value={form.agent_name} onChange={(e) => setForm({ ...form, agent_name: e.target.value })}>
          <option value="">all agents</option>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </Field>
      <Field label="Recent window (hours)">
        <input className="input" type="number" min={1} value={form.recent_hours} onChange={(e) => setForm({ ...form, recent_hours: Number(e.target.value) || 1 })} />
      </Field>
      <Field label="Baseline window (hours)">
        <input className="input" type="number" min={1} value={form.baseline_hours} onChange={(e) => setForm({ ...form, baseline_hours: Number(e.target.value) || 1 })} />
      </Field>
      <div className="flex flex-col justify-end text-[13px]">
        <label className="flex items-center gap-2"><input type="checkbox" checked={form.production_only} onChange={(e) => setForm({ ...form, production_only: e.target.checked })} /> production rollouts only</label>
      </div>
      <div className="flex items-end">
        <button className="btn btn-primary w-full" disabled={busy}>{busy ? "Analysing…" : "Run analysis"}</button>
      </div>
      {err ? <div className="md:col-span-5"><ErrorBox error={err} /></div> : null}
    </form>
  );
}

function DriftPanel({ drift }: { drift: IntelligenceReport["drift"] }) {
  const metrics = Object.entries(drift.metrics || {});
  return (
    <Panel title={`Drift alerts · ${drift.alerts.length}`} actions={<span className="text-[11px] text-[var(--muted)]">baseline n={drift.n_baseline} · recent n={drift.n_recent}</span>}>
      <div className="space-y-4">
        {drift.alerts.length === 0 ? (
          <div className="flex items-center gap-2 text-[var(--muted)]">
            <Chip status="ok">no alerts</Chip>
            {drift.note && <span>{drift.note}</span>}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Type</th><th>Metric</th><th>Severity</th><th>Message</th><th>Δ / effect</th><th>p</th></tr></thead>
              <tbody>
                {drift.alerts.map((a, i) => (
                  <tr key={i}>
                    <td><Chip status={severityChip[a.severity || ""] || "unset"}>{a.type}</Chip></td>
                    <td className="mono text-[12px]">{a.metric ?? a.family ?? "–"}</td>
                    <td>{a.severity || "–"}</td>
                    <td>{a.message || "–"}</td>
                    <td className="tabular-nums">{a.delta !== undefined ? delta(a.delta) : a.ratio !== undefined ? `${num(a.ratio)}x` : a.value !== undefined ? num(a.value) : "–"}</td>
                    <td className="tabular-nums">{a.p_value !== undefined ? pval(a.p_value) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {metrics.length > 0 && (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Metric</th><th>Baseline</th><th>Recent</th><th>Δ</th><th>p-value</th></tr></thead>
              <tbody>
                {metrics.map(([k, m]) => (
                  <tr key={k}>
                    <td className="mono text-[12px]">{k}</td>
                    <td className="tabular-nums">{pct(m.baseline)}</td>
                    <td className="tabular-nums">{pct(m.recent)}</td>
                    <td className={`tabular-nums ${m.delta > 0 ? "text-emerald-300" : m.delta < 0 ? "text-rose-300" : ""}`}>{delta(m.delta)}</td>
                    <td className={`tabular-nums ${m.p_value < 0.05 ? "font-semibold" : "text-[var(--muted)]"}`}>{pval(m.p_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Panel>
  );
}

function FailuresPanel({ failures }: { failures: IntelligenceReport["failures"] }) {
  const trendCell = (c: FailureCluster) => {
    if (c.trend === null || c.trend === undefined) return <span className="text-[var(--muted)]">–</span>;
    return <span className={`tabular-nums ${c.trend > 0 ? "text-rose-300" : c.trend < 0 ? "text-emerald-300" : ""}`}>{delta(c.trend)}</span>;
  };
  return (
    <Panel title={`Failure clusters · ${failures.clusters.length}`} actions={<span className="text-[11px] text-[var(--muted)]">{failures.n_failed} / {failures.n_rollouts} rollouts failed · {pct(failures.failure_rate)}</span>}>
      {failures.clusters.length === 0 ? (
        <div className="text-[var(--muted)]">No failure clusters.</div>
      ) : (
        <div className="max-h-[28rem] overflow-auto">
          <table className="tbl">
            <thead><tr><th>Kind</th><th>Role</th><th>Tool</th><th>Family</th><th>Count</th><th>Rate</th><th>Baseline</th><th>Trend</th><th>Examples</th></tr></thead>
            <tbody>
              {failures.clusters.map((c, i) => (
                <tr key={i}>
                  <td><Chip status={c.kind === "tool_error" ? "failed" : c.kind === "wrong_tool" ? "4xx" : "unset"}>{c.kind}</Chip></td>
                  <td>{c.role}</td>
                  <td className="mono text-[12px]">{c.tool === "-" ? <span className="text-[var(--muted)]">–</span> : c.tool}</td>
                  <td>{c.family}</td>
                  <td className="tabular-nums font-medium">{c.count}</td>
                  <td className="tabular-nums">{pct(c.rate)}</td>
                  <td className="tabular-nums">{pct(c.baseline_rate)}</td>
                  <td>{trendCell(c)}</td>
                  <td>
                    <div className="flex flex-wrap gap-x-2 gap-y-0.5">
                      {c.examples.slice(0, 3).map((id) => <IdLink key={id} href={`/rollouts/detail?id=${id}`} id={id} n={12} />)}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function CoveragePanel({ coverage, onMined }: { coverage: IntelligenceReport["coverage"]; onMined: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [mined, setMined] = useState<MineResult | null>(null);
  const [form, setForm] = useState({ name: "mined from production", since_hours: 720, production_only: false });
  async function mine() {
    setBusy(true);
    setErr(null);
    setMined(null);
    try {
      setMined(await Api.mineTasks(form));
      onMined();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Panel title="Eval coverage of production traffic">
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <KPI label="Uncovered traffic" value={pct(coverage.uncovered_traffic_share)} hint="share of production rollouts with no matching eval task" />
          <KPI label="Uncovered patterns" value={coverage.uncovered_patterns} hint={coverage.production_patterns !== undefined ? `of ${coverage.production_patterns} tool patterns` : undefined} />
          <KPI label="Covered patterns" value={coverage.covered_patterns ?? "–"} />
          <KPI label="Minable tasks" value={coverage.minable_tasks ?? "–"} hint="successful rollouts convertible to eval tasks" />
        </div>
        {coverage.top_uncovered && coverage.top_uncovered.length > 0 && (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Uncovered tool pattern</th><th>Rollouts</th></tr></thead>
              <tbody>
                {coverage.top_uncovered.map((p, i) => (
                  <tr key={i}>
                    <td className="mono text-[12px]">{p.tools.join(" → ")}</td>
                    <td className="tabular-nums">{p.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {coverage.tools_never_evaluated && coverage.tools_never_evaluated.length > 0 && (
          <div className="text-[12px] text-[var(--muted)]">
            Tools never evaluated: {coverage.tools_never_evaluated.map((t) => <code key={t} className="mono mr-1.5 rounded bg-[var(--bg)] px-1 py-0.5 text-[11px] text-[var(--fg)]">{t}</code>)}
          </div>
        )}
        <div className="flex flex-wrap items-end gap-2 border-t border-[var(--border)] pt-3">
          <Field label="Dataset name">
            <input className="input w-56" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </Field>
          <Field label="Since (hours)">
            <input className="input w-28" type="number" min={1} value={form.since_hours} onChange={(e) => setForm({ ...form, since_hours: Number(e.target.value) || 1 })} />
          </Field>
          <label className="flex items-center gap-2 pb-2 text-[13px]"><input type="checkbox" checked={form.production_only} onChange={(e) => setForm({ ...form, production_only: e.target.checked })} /> production only</label>
          <button type="button" className="btn btn-primary" disabled={busy || !form.name.trim()} onClick={mine}>{busy ? "Mining…" : "Mine tasks from production"}</button>
          {mined && (
            <span className="text-[12px] text-emerald-300">
              Mined {mined.n_tasks} tasks from {mined.from_rollouts} rollouts → <Link href="/datasets" className="link mono">{mined.dataset_id}</Link>
            </span>
          )}
        </div>
        {err ? <ErrorBox error={err} /> : null}
      </div>
    </Panel>
  );
}

function RecommendationsPanel({ recs }: { recs: Recommendation[] }) {
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const { data: datasets } = useSWR<Dataset[]>("/v1/datasets", fetcher);
  const [agentId, setAgentId] = useState("");
  const [trainDs, setTrainDs] = useState("");
  const [evalDs, setEvalDs] = useState("");
  const [busy, setBusy] = useState<number | null>(null);
  const [results, setResults] = useState<Record<number, ApplyResult>>({});
  const [errs, setErrs] = useState<Record<number, unknown>>({});
  const agentsDeployedFirst = [...(agents || [])].sort((a, b) => (a.status === "deployed" ? -1 : 0) - (b.status === "deployed" ? -1 : 0));

  async function apply(i: number, rec: Recommendation) {
    setBusy(i);
    setErrs({ ...errs, [i]: null });
    try {
      const r = await Api.applyRecommendation({ recommendation: rec, agent_id: agentId || null, train_dataset_id: trainDs || null, eval_dataset_id: evalDs || null });
      setResults({ ...results, [i]: r });
    } catch (x) {
      setErrs({ ...errs, [i]: x });
    } finally {
      setBusy(null);
    }
  }
  const actionable = (r: Recommendation) => ["training", "attribution", "mine_tasks"].includes(r.action?.kind);
  return (
    <Panel title={`Recommendations · ${recs.length}`}>
      <div className="space-y-3">
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Agent for training / attribution">
            <select className="input" value={agentId} onChange={(e) => setAgentId(e.target.value)}>
              <option value="">default (deployed agent)</option>
              {agentsDeployedFirst.map((a) => <option key={a.id} value={a.id}>{a.name}/{a.version} ({a.status})</option>)}
            </select>
          </Field>
          <Field label="Train dataset">
            <select className="input" value={trainDs} onChange={(e) => setTrainDs(e.target.value)}>
              <option value="">default (latest train split)</option>
              {(datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name} · {d.split} · {d.n_tasks}</option>)}
            </select>
          </Field>
          <Field label="Eval dataset">
            <select className="input" value={evalDs} onChange={(e) => setEvalDs(e.target.value)}>
              <option value="">default (latest dataset)</option>
              {(datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name} · {d.split} · {d.n_tasks}</option>)}
            </select>
          </Field>
        </div>
        {recs.length === 0 ? (
          <div className="text-[var(--muted)]">No recommendations — production looks healthy.</div>
        ) : (
          <div className="space-y-2">
            {recs.map((r, i) => {
              const res = results[i];
              return (
                <div key={i} className="rounded-md border border-[var(--border)] bg-[var(--panel-2)] px-3 py-2.5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="tabular-nums text-[12px] text-[var(--muted)]">P {num(r.priority, 2)}</span>
                        <Chip status={r.action?.kind === "training" ? "agent" : r.action?.kind === "mine_tasks" ? "candidate" : r.action?.kind === "attribution" ? "llm" : "unset"}>{r.action?.kind || "info"}</Chip>
                        {r.action?.algorithm && <span className="mono text-[11px] text-[var(--muted)]">{r.action.algorithm}</span>}
                        <span className="font-medium">{r.title}</span>
                      </div>
                      <div className="mt-0.5 text-[12px] text-[var(--muted)]">{r.why}</div>
                      {(r.action?.params || r.action?.what) && (
                        <div className="mono mt-1 text-[11px] text-[var(--muted)]">{r.action.what ?? JSON.stringify(r.action.params)}</div>
                      )}
                      {res && (
                        <div className="mt-1.5 text-[12px] text-emerald-300">
                          Started <span className="mono">{res.kind}</span>
                          {res.training_run_id && <> · <Link href={`/training/detail?id=${res.training_run_id}`} className="link mono">{res.training_run_id}</Link></>}
                          {res.job_id && <> · job <Link href={`/jobs?id=${res.job_id}`} className="link mono">{res.job_id}</Link></>}
                          {res.dataset_id && <> · {res.n_tasks} tasks → <Link href="/datasets" className="link mono">{res.dataset_id}</Link></>}
                        </div>
                      )}
                      {errs[i] ? <div className="mt-1.5"><ErrorBox error={errs[i]} /></div> : null}
                    </div>
                    {actionable(r) && (
                      <button type="button" className="btn btn-primary" disabled={busy !== null} onClick={() => apply(i, r)}>{busy === i ? "Applying…" : "Apply"}</button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Panel>
  );
}

function ReportView({ report, onMined }: { report: IntelligenceReport; onMined: () => void }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <KPI label="Recent rollouts" value={report.n_recent} hint={`last ${report.recent_hours}h`} />
        <KPI label="Baseline rollouts" value={report.n_baseline} hint={`previous ${report.baseline_hours}h`} />
        <KPI label="Health" value={<Chip status={report.healthy ? "ok" : "failed"}>{report.healthy ? "healthy" : "unhealthy"}</Chip>} hint={`${report.drift.alerts.length} drift alerts`} />
        <KPI label="Failure rate" value={pct(report.failures.failure_rate)} hint={`${report.failures.n_failed} of ${report.failures.n_rollouts} rollouts`} />
        <KPI label="Uncovered traffic" value={pct(report.coverage.uncovered_traffic_share)} hint={`${report.coverage.uncovered_patterns} patterns`} />
      </div>
      <DriftPanel drift={report.drift} />
      <FailuresPanel failures={report.failures} />
      <CoveragePanel coverage={report.coverage} onMined={onMined} />
      <RecommendationsPanel recs={report.recommendations || []} />
    </div>
  );
}

export default function IntelligencePage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const { data: reports, error: reportsErr, mutate: mutateReports } = useSWR<IntelligenceReportSummary[]>(q("/v1/intelligence/reports", { limit: 50 }), fetcher);
  const key = selected ? `/v1/intelligence/reports/${selected}` : "/v1/intelligence/latest";
  const { data: report, error, mutate } = useSWR<IntelligenceReport>(key, fetcher, { shouldRetryOnError: false });
  const noReport = error instanceof ApiError && error.status === 404;

  function onRun(r: IntelligenceReport) {
    setSelected(null);
    setShowForm(false);
    mutate(r, { revalidate: false });
    mutateReports();
  }

  return (
    <>
      <PageHeader
        title="Production intelligence"
        subtitle={report ? <span className="flex flex-wrap items-center gap-2">report <span className="mono">{report.id}</span> · {report.agent_name ? <>agent <span className="mono">{report.agent_name}</span></> : "all agents"} · {ts(report.created_at)}{selected && <Chip status="candidate">historical</Chip>}</span> : "Drift, failure clusters, eval coverage and recommended actions from production rollouts."}
        actions={<button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "Run analysis"}</button>}
      />
      {(showForm || noReport) && (
        <Panel title="Run analysis" className="mb-4">
          <RunForm onDone={onRun} />
        </Panel>
      )}
      {noReport ? (
        <div className="mb-4 rounded-md border border-[var(--border)] bg-[var(--panel)] px-4 py-6 text-center text-[var(--muted)]">No intelligence report yet — run an analysis above.</div>
      ) : (
        <DataState data={report} error={error}>
          {(r) => <ReportView report={r} onMined={() => {}} />}
        </DataState>
      )}
      <div className="mt-4">
        <Panel title="Report history">
          <DataState data={reports} error={reportsErr} isEmpty={(d) => d.length === 0} empty="No reports yet.">
            {(rows) => (
              <div className="overflow-x-auto">
                <table className="tbl">
                  <thead><tr><th>Report</th><th>Agent</th><th>Windows</th><th>Recent</th><th>Baseline</th><th>Health</th><th>Alerts</th><th>Recommendations</th><th>Created</th></tr></thead>
                  <tbody>
                    {rows.map((r) => {
                      const active = report?.id === r.id;
                      return (
                        <tr key={r.id} className={`cursor-pointer ${active ? "bg-[var(--panel-2)]" : ""}`} onClick={() => setSelected(r.id)}>
                          <td className="mono text-[12px]">{active && <span className="mr-1 text-[var(--accent)]">▸</span>}{r.id}</td>
                          <td>{r.agent_name || <span className="text-[var(--muted)]">all</span>}</td>
                          <td className="tabular-nums text-[var(--muted)]">{r.recent_hours}h vs {r.baseline_hours}h</td>
                          <td className="tabular-nums">{r.n_recent}</td>
                          <td className="tabular-nums">{r.n_baseline}</td>
                          <td><Chip status={r.healthy ? "ok" : "failed"}>{r.healthy ? "healthy" : "unhealthy"}</Chip></td>
                          <td className="tabular-nums">{r.n_alerts}</td>
                          <td className="tabular-nums">{r.n_recommendations}</td>
                          <td className="whitespace-nowrap text-[var(--muted)]">{ts(r.created_at)}</td>
                        </tr>
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
