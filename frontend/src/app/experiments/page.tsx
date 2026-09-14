"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { Api, fetcher, type Agent, type Experiment } from "@/lib/api";
import { delta, isNum, pct, pval, ts } from "@/lib/format";
import { MetricBarChart } from "@/components/charts";
import { Chip, DataState, ErrorBox, Field, PageHeader, Panel } from "@/components/ui";

type VariantResult = { n: number; mean: number; ci: [number, number] };

function NewExperimentForm({ agents, onDone }: { agents: Agent[]; onDone: () => void }) {
  const [name, setName] = useState("");
  const [vars, setVars] = useState([
    { name: "control", agent_id: "", weight: 1 },
    { name: "treatment", agent_id: "", weight: 1 },
  ]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const deployed = agents.find((a) => a.status === "deployed")?.id || agents[0]?.id || "";
  const other = agents.find((a) => a.id !== deployed)?.id || deployed;
  const resolved = vars.map((v, i) => ({ ...v, agent_id: v.agent_id || (i === 0 ? deployed : other) }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await Api.createExperiment({ name, variants: resolved });
      onDone();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  const upd = (i: number, patch: Partial<(typeof vars)[number]>) => setVars(vars.map((v, j) => (j === i ? { ...v, ...patch } : v)));
  return (
    <form onSubmit={submit} className="space-y-3">
      <Field label="Experiment name">
        <input className="input md:w-96" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. v3 vs v0 on live traffic" required />
      </Field>
      <div className="grid gap-3 md:grid-cols-2">
        {resolved.map((v, i) => (
          <div key={i} className="grid grid-cols-[1fr_2fr_5rem] gap-2 rounded-md border border-[var(--border)] p-3">
            <Field label="Variant"><input className="input" value={v.name} onChange={(e) => upd(i, { name: e.target.value })} required /></Field>
            <Field label="Agent">
              <select className="input" value={v.agent_id} onChange={(e) => upd(i, { agent_id: e.target.value })}>
                {agents.map((a) => <option key={a.id} value={a.id}>{a.name}/{a.version} ({a.status})</option>)}
              </select>
            </Field>
            <Field label="Weight"><input className="input" type="number" min={0} step={0.1} value={v.weight} onChange={(e) => upd(i, { weight: Number(e.target.value) })} /></Field>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-3">
        <button className="btn btn-primary" disabled={busy || !name || agents.length === 0}>{busy ? "Creating…" : "Create experiment"}</button>
        {err ? <ErrorBox error={err} /> : null}
      </div>
    </form>
  );
}

function ExperimentCard({ e, agents, onChange }: { e: Experiment; agents: Agent[]; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const results = e.results || {};
  const rows = e.variants.map((v) => {
    const r = results[v.name] as VariantResult | undefined;
    return { name: v.name, agent_id: v.agent_id, weight: v.weight, n: r?.n, mean: r?.mean, ci: r?.ci };
  });
  const agentLabel = (id: string) => {
    const a = agents.find((x) => x.id === id);
    return a ? `${a.name}/${a.version}` : id;
  };
  const chart = rows.filter((r) => isNum(r.mean)).map((r) => ({ name: r.name, mean: r.mean }));
  return (
    <div className="panel p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="text-[15px] font-semibold">{e.name}</div>
          <div className="mono text-[11px] text-[var(--muted)]">{e.id} · {ts(e.created_at)}</div>
        </div>
        <div className="flex items-center gap-2">
          <Chip status={e.status} />
          {e.status === "running" && (
            <button className="btn" disabled={busy} onClick={async () => { setBusy(true); try { await Api.stopExperiment(e.id); onChange(); } finally { setBusy(false); } }}>Stop</button>
          )}
        </div>
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-[1fr_16rem]">
        <table className="tbl">
          <thead><tr><th>Variant</th><th>Agent</th><th>Weight</th><th>n</th><th>Mean</th><th>95% CI</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td className="font-medium">{r.name}</td>
                <td><Link href={`/agents/detail?id=${r.agent_id}`} className="link">{agentLabel(r.agent_id)}</Link></td>
                <td className="tabular-nums">{r.weight}</td>
                <td className="tabular-nums">{r.n ?? "–"}</td>
                <td className="tabular-nums font-medium">{pct(r.mean)}</td>
                <td className="tabular-nums text-[var(--muted)]">{r.ci ? `[${pct(r.ci[0])}, ${pct(r.ci[1])}]` : "–"}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={6} className="text-[12px] text-[var(--muted)]">
                {e.n_outcomes} outcomes · delta{" "}
                <span className={`font-medium ${isNum(results.delta) ? (results.delta > 0 ? "text-emerald-300" : results.delta < 0 ? "text-rose-300" : "") : ""}`}>{delta(results.delta)}</span> · p-value{" "}
                <span className={isNum(results.p_value) && results.p_value < 0.05 ? "font-semibold text-[var(--fg)]" : ""}>{pval(results.p_value)}</span>
                {isNum(results.p_value) && results.p_value < 0.05 && <span className="ml-1 text-emerald-300">significant</span>}
              </td>
            </tr>
          </tfoot>
        </table>
        <div>{chart.length ? <MetricBarChart data={chart} xKey="name" series={[{ key: "mean", label: "mean outcome" }]} height={170} /> : <div className="text-[12px] text-[var(--muted)]">No outcomes yet.</div>}</div>
      </div>
    </div>
  );
}

export default function ExperimentsPage() {
  const { data, error, mutate } = useSWR<Experiment[]>("/v1/experiments", fetcher, { refreshInterval: 5000 });
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const [showForm, setShowForm] = useState(false);
  return (
    <>
      <PageHeader title="Experiments" subtitle="Online A/B comparisons between agent versions" actions={<button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "New experiment"}</button>} />
      {showForm && (
        <Panel title="New experiment" className="mb-4">
          <NewExperimentForm agents={agents || []} onDone={() => { mutate(); setShowForm(false); }} />
        </Panel>
      )}
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No experiments yet.">
        {(rows) => (
          <div className="space-y-4">
            {rows.map((e) => <ExperimentCard key={e.id} e={e} agents={agents || []} onChange={() => mutate()} />)}
            <div className="text-[11px] text-[var(--muted)]">Assignments: <span className="mono">GET /v1/experiments/{"{id}"}/assign?unit=…</span> · outcomes: <span className="mono">POST /v1/experiments/{"{id}"}/outcomes</span></div>
          </div>
        )}
      </DataState>
    </>
  );
}
