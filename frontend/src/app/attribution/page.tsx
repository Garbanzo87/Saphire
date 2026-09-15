"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import useSWR from "swr";
import { Api, fetcher, type Agent, type AttributionJob, type AttributionResult, type Dataset } from "@/lib/api";
import { isActive, ms, pct, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { MetricBarChart } from "@/components/charts";
import { Chip, DataState, ErrorBox, Expander, Field, IdLink, KPI, Logs, PageHeader, Panel, Progress } from "@/components/ui";

function RunForm({ onCreated }: { onCreated: (j: AttributionJob) => void }) {
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const { data: datasets } = useSWR<Dataset[]>("/v1/datasets", fetcher);
  const multi = (agents || []).filter((a) => a.config?.roles && Object.keys(a.config.roles).length > 0);
  const [form, setForm] = useState({ agent_id: "", dataset_id: "", reference_model: "mock", degraded_model: "mock:error=0.9", shapley: true, n_permutations: 4 });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const agentId = form.agent_id || multi[0]?.id || "";
  const datasetId = form.dataset_id || datasets?.find((d) => d.split === "eval")?.id || datasets?.[0]?.id || "";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const job = await Api.createAttribution({
        agent_id: agentId,
        dataset_id: datasetId,
        reference_model: form.reference_model.trim() || null,
        degraded_model: form.degraded_model.trim() || null,
        shapley: form.shapley,
        n_permutations: form.n_permutations,
      });
      onCreated(job);
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form onSubmit={submit} className="grid gap-3 md:grid-cols-6">
      <Field label="Agent (multi-agent only)">
        <select className="input" value={agentId} onChange={(e) => setForm({ ...form, agent_id: e.target.value })}>
          {multi.length === 0 && <option value="">no multi-agent agents</option>}
          {multi.map((a) => <option key={a.id} value={a.id}>{a.name}/{a.version} ({a.status}) · {Object.keys(a.config.roles || {}).length} roles</option>)}
        </select>
      </Field>
      <Field label="Dataset">
        <select className="input" value={datasetId} onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}>
          {(datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name} · {d.split} · {d.n_tasks}</option>)}
        </select>
      </Field>
      <Field label="Reference model (upgrade)">
        <input className="input" value={form.reference_model} onChange={(e) => setForm({ ...form, reference_model: e.target.value })} placeholder="mock" />
      </Field>
      <Field label="Degraded model">
        <input className="input" value={form.degraded_model} onChange={(e) => setForm({ ...form, degraded_model: e.target.value })} placeholder="mock:error=0.9" />
      </Field>
      <div className="flex flex-col gap-1.5">
        <Field label="Permutations">
          <input className="input" type="number" min={1} max={64} value={form.n_permutations} disabled={!form.shapley} onChange={(e) => setForm({ ...form, n_permutations: Number(e.target.value) || 1 })} />
        </Field>
        <label className="flex items-center gap-2 text-[13px]"><input type="checkbox" checked={form.shapley} onChange={(e) => setForm({ ...form, shapley: e.target.checked })} /> Shapley values</label>
      </div>
      <div className="flex items-end">
        <button className="btn btn-primary w-full" disabled={busy || !agentId || !datasetId || !(form.reference_model.trim() || form.degraded_model.trim())}>{busy ? "Starting…" : "Run attribution"}</button>
      </div>
      {err ? <div className="md:col-span-6"><ErrorBox error={err} /></div> : null}
    </form>
  );
}

function ResultView({ r }: { r: AttributionResult }) {
  const roles = Object.entries(r.ablation?.roles || {});
  const ablationRows = roles.map(([role, v]) => ({ role, headroom: v.headroom, criticality: v.criticality }));
  const shap = Object.entries(r.shapley?.shapley || {});
  const shapRows = shap.map(([role, v]) => ({ role, shapley: v, share: r.shapley?.share?.[role] ?? 0 }));
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <KPI label="Baseline success" value={pct(r.ablation?.baseline ?? r.shapley?.baseline)} hint={`${r.n_tasks} tasks`} />
        <KPI label="All roles upgraded" value={pct(r.ablation?.all_upgraded ?? r.shapley?.all_upgraded)} hint={r.ablation?.reference_model ? <>ref <span className="mono">{r.ablation.reference_model}</span></> : undefined} />
        <KPI label="Total headroom" value={pct(r.ablation?.total_headroom)} hint="upgrade everything − baseline" />
        <KPI label="Roles" value={roles.length} hint={r.ablation?.degraded_model ? <>degraded <span className="mono">{r.ablation.degraded_model}</span></> : undefined} />
        <KPI label="Shapley evaluations" value={r.shapley?.n_evaluations ?? "–"} hint={r.shapley ? `${r.shapley.n_permutations} permutations` : "shapley off"} />
      </div>
      {r.recommendation && (
        <div className="flex flex-wrap items-center gap-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2.5 text-[13px]">
          <Chip status="ok">recommendation</Chip>
          <span>
            Optimise {r.recommendation.optimize_roles.map((x) => <code key={x} className="mono mr-1 rounded bg-[var(--bg)] px-1 py-0.5 text-[12px]">{x}</code>)}
          </span>
          <span className="text-[var(--muted)]">— {r.recommendation.reason}</span>
        </div>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Role ablation · headroom & criticality">
          {roles.length === 0 ? (
            <div className="text-[var(--muted)]">No ablation data.</div>
          ) : (
            <div className="space-y-3">
              <MetricBarChart data={ablationRows} xKey="role" series={[{ key: "headroom", label: "headroom (upgrade gain)" }, { key: "criticality", label: "criticality (degrade loss)" }]} height={220} />
              <div className="overflow-x-auto">
                <table className="tbl">
                  <thead><tr><th>Role</th><th>Upgraded</th><th>Headroom</th><th>Degraded</th><th>Criticality</th></tr></thead>
                  <tbody>
                    {roles.sort((a, b) => b[1].headroom - a[1].headroom).map(([role, v]) => (
                      <tr key={role}>
                        <td className="font-medium">{role}</td>
                        <td className="tabular-nums">{pct(v.upgraded)}</td>
                        <td className={`tabular-nums ${v.headroom > 0 ? "text-emerald-300" : ""}`}>{pct(v.headroom)}</td>
                        <td className="tabular-nums">{pct(v.degraded)}</td>
                        <td className={`tabular-nums ${v.criticality > 0 ? "text-rose-300" : ""}`}>{pct(v.criticality)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Panel>
        <Panel title="Shapley value of upgrading each role">
          {shap.length === 0 ? (
            <div className="text-[var(--muted)]">Shapley not computed for this job.</div>
          ) : (
            <div className="space-y-3">
              <MetricBarChart data={shapRows} xKey="role" series={[{ key: "shapley", label: "shapley (success gain)" }]} height={220} />
              <div className="overflow-x-auto">
                <table className="tbl">
                  <thead><tr><th>Role</th><th>Shapley</th><th>Share of headroom</th></tr></thead>
                  <tbody>
                    {shapRows.sort((a, b) => b.shapley - a.shapley).map((s) => (
                      <tr key={s.role}>
                        <td className="font-medium">{s.role}</td>
                        <td className={`tabular-nums ${s.shapley > 0 ? "text-emerald-300" : s.shapley < 0 ? "text-rose-300" : ""}`}>{pct(s.shapley)}</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-28 overflow-hidden rounded bg-[var(--border)]"><div className="h-full bg-[var(--accent)]" style={{ width: `${Math.max(0, Math.min(1, s.share)) * 100}%` }} /></div>
                            <span className="tabular-nums text-[12px] text-[var(--muted)]">{pct(s.share, 0)}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

export default function AttributionPage() {
  const { data, error, mutate } = useSWR<AttributionJob[]>("/v1/attribution", fetcher, { refreshInterval: (d) => (d?.some((j) => isActive(j.status)) ? 3000 : 0) });
  const [selected, setSelected] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [pollErr, setPollErr] = useState<unknown>(null);
  const job = (data || []).find((j) => j.id === selected) || data?.[0];

  // Poll a freshly-created job until it finishes, then refresh the list.
  useEffect(() => {
    if (!job || !isActive(job.status)) return;
    let stop = false;
    const t = setInterval(async () => {
      try {
        const j = await Api.job(job.id);
        if (!isActive(j.status) && !stop) {
          clearInterval(t);
          mutate();
        }
      } catch (x) {
        setPollErr(x);
      }
    }, 2000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [job, mutate]);

  return (
    <>
      <PageHeader
        title="Multi-agent attribution"
        subtitle={data ? `${data.length} attribution jobs · counterfactual role ablation and Shapley credit` : "Which role to optimise next: counterfactual role ablation and Shapley credit."}
        actions={<button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "Run attribution"}</button>}
      />
      {showForm && (
        <Panel title="Run attribution" className="mb-4">
          <RunForm
            onCreated={(j) => {
              setShowForm(false);
              setSelected(j.id);
              mutate();
            }}
          />
        </Panel>
      )}
      {pollErr ? <div className="mb-3"><ErrorBox error={pollErr} /></div> : null}
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No attribution jobs yet — run one above.">
        {(rows) => (
          <div className="space-y-4">
            <div className="panel overflow-x-auto">
              <table className="tbl">
                <thead><tr><th>Job</th><th>Agent</th><th>Dataset</th><th>Reference</th><th>Degraded</th><th>Shapley</th><th>Status</th><th>Baseline</th><th>Headroom</th><th>Optimise</th><th>Duration</th><th>Created</th></tr></thead>
                <tbody>
                  {rows.map((j) => {
                    const p = j.params as Record<string, unknown>;
                    const active = job?.id === j.id;
                    const dur = j.started_at ? ((j.finished_at ?? Date.now() / 1000) - j.started_at) * 1000 : null;
                    return (
                      <tr key={j.id} className={`cursor-pointer ${active ? "bg-[var(--panel-2)]" : ""}`} onClick={() => setSelected(j.id)}>
                        <td className="mono text-[12px]">{active && <span className="mr-1 text-[var(--accent)]">▸</span>}{j.id}</td>
                        <td><Link href={`/agents/detail?id=${p.agent_id}`} className="hover:underline" onClick={(e) => e.stopPropagation()}>{String(p.agent_name ?? p.agent_id)}{p.agent_version ? <span className="mono text-[var(--muted)]"> /{String(p.agent_version)}</span> : null}</Link></td>
                        <td>{String(p.dataset_name ?? p.dataset_id ?? "–")}</td>
                        <td className="mono text-[12px]">{String(p.reference_model ?? "–")}</td>
                        <td className="mono text-[12px]">{String(p.degraded_model ?? "–")}</td>
                        <td>{p.shapley ? <span className="tabular-nums">{String(p.n_permutations ?? "")} perm</span> : <span className="text-[var(--muted)]">off</span>}</td>
                        <td><Chip status={j.status} /></td>
                        <td className="tabular-nums">{pct(j.result?.ablation?.baseline)}</td>
                        <td className="tabular-nums">{pct(j.result?.ablation?.total_headroom)}</td>
                        <td>{j.result?.recommendation?.optimize_roles?.join(", ") || <span className="text-[var(--muted)]">–</span>}</td>
                        <td className="tabular-nums">{ms(dur)}</td>
                        <td className="whitespace-nowrap text-[var(--muted)]">{ts(j.created_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {job && (
              <>
                <div className="flex flex-wrap items-center gap-2 text-[13px]">
                  <span className="text-[var(--muted)]">Selected</span> <span className="mono">{job.id}</span> <Chip status={job.status} />
                  {isActive(job.status) && <><Progress value={job.progress} /><span className="text-[var(--muted)]">polling…</span></>}
                  <Link href={`/jobs?id=${job.id}`} className="link text-[12px]">job →</Link>
                  {job.result?.dataset_id && <span className="text-[var(--muted)]">dataset <IdLink href="/datasets" id={job.result.dataset_id} /></span>}
                </div>
                {job.error && <ErrorBox error={job.error} />}
                {job.result ? <ResultView r={job.result} /> : !isActive(job.status) && !job.error ? <div className="text-[var(--muted)]">No result.</div> : null}
                <Expander title="Job logs"><Logs logs={job.logs} /></Expander>
                <Expander title="Result JSON"><JsonView value={job.result} /></Expander>
              </>
            )}
          </div>
        )}
      </DataState>
    </>
  );
}
