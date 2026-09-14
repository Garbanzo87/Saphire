"use client";
import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";
import { Api, fetcher, q, type Agent, type Dataset, type EvalRun } from "@/lib/api";
import { ms, num, pct, ts } from "@/lib/format";
import { Chip, DataState, ErrorBox, Field, IdLink, PageHeader, Panel } from "@/components/ui";

function passHatK(m: EvalRun["metrics"], k: number) {
  if (!m) return undefined;
  return m[`pass_hat_${k}`] ?? m[`pass_at_${k}`] ?? m.pass_at_1;
}

function NewEvalForm({ onCreated }: { onCreated: (e: EvalRun) => void }) {
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const { data: datasets } = useSWR<Dataset[]>("/v1/datasets", fetcher);
  const [form, setForm] = useState({ agent_id: "", dataset_id: "", k: 1, judge_model: "", gate: false, auto_promote: false });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const agentId = form.agent_id || agents?.[0]?.id || "";
  const datasetId = form.dataset_id || datasets?.find((d) => d.split === "eval")?.id || datasets?.[0]?.id || "";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const run = await Api.createEval({ agent_id: agentId, dataset_id: datasetId, k: form.k, judge_model: form.judge_model || null, gate: form.gate, auto_promote: form.auto_promote });
      onCreated(run);
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form onSubmit={submit} className="grid gap-3 md:grid-cols-6">
      <Field label="Agent">
        <select className="input" value={agentId} onChange={(e) => setForm({ ...form, agent_id: e.target.value })}>
          {(agents || []).map((a) => <option key={a.id} value={a.id}>{a.name}/{a.version} ({a.status})</option>)}
        </select>
      </Field>
      <Field label="Dataset">
        <select className="input" value={datasetId} onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}>
          {(datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name} · {d.split} · {d.n_tasks}</option>)}
        </select>
      </Field>
      <Field label="k (trials / task)">
        <input className="input" type="number" min={1} max={16} value={form.k} onChange={(e) => setForm({ ...form, k: Number(e.target.value) || 1 })} />
      </Field>
      <Field label="Judge model">
        <select className="input" value={form.judge_model} onChange={(e) => setForm({ ...form, judge_model: e.target.value })}>
          <option value="">none</option>
          <option value="mock">mock</option>
        </select>
      </Field>
      <div className="flex flex-col justify-end gap-1.5 text-[13px]">
        <label className="flex items-center gap-2"><input type="checkbox" checked={form.gate} onChange={(e) => setForm({ ...form, gate: e.target.checked })} /> gate vs deployed</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={form.auto_promote} disabled={!form.gate} onChange={(e) => setForm({ ...form, auto_promote: e.target.checked })} /> auto-promote</label>
      </div>
      <div className="flex items-end">
        <button className="btn btn-primary w-full" disabled={busy || !agentId || !datasetId}>{busy ? "Starting…" : "Run evaluation"}</button>
      </div>
      {err ? <div className="md:col-span-6"><ErrorBox error={err} /></div> : null}
    </form>
  );
}

export default function EvalsPage() {
  const router = useRouter();
  const { data, error, mutate } = useSWR<EvalRun[]>(q("/v1/evals", { limit: 100 }), fetcher, { refreshInterval: 5000 });
  const [showForm, setShowForm] = useState(false);
  return (
    <>
      <PageHeader title="Evaluations" subtitle={data ? `${data.length} eval runs` : undefined} actions={<button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "New evaluation"}</button>} />
      {showForm && (
        <Panel title="New evaluation" className="mb-4">
          <NewEvalForm onCreated={(e) => { mutate(); router.push(`/evals/detail?id=${e.id}`); }} />
        </Panel>
      )}
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No eval runs yet — start one above.">
        {(rows) => (
          <div className="panel overflow-x-auto">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Agent</th>
                  <th>Suite</th>
                  <th>k</th>
                  <th>Judge</th>
                  <th>Status</th>
                  <th>Task success</th>
                  <th>Tool F1</th>
                  <th>Context</th>
                  <th>pass^k</th>
                  <th>p95 latency</th>
                  <th>Throughput</th>
                  <th>n</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((e) => (
                  <tr key={e.id}>
                    <td><IdLink href={`/evals/detail?id=${e.id}`} id={e.id} n={20} /></td>
                    <td>
                      <Link href={`/agents/detail?id=${e.agent_id}`} className="hover:underline">{e.agent_name ?? e.agent_id}{e.agent_version ? <span className="mono text-[var(--muted)]"> /{e.agent_version}</span> : null}</Link>
                    </td>
                    <td>{e.suite}</td>
                    <td className="tabular-nums">{e.k}</td>
                    <td>{e.judge_model || <span className="text-[var(--muted)]">none</span>}</td>
                    <td><Chip status={e.status} /></td>
                    <td className="tabular-nums font-medium">{pct(e.metrics?.task_success)}</td>
                    <td className="tabular-nums">{pct(e.metrics?.tool_selection_f1)}</td>
                    <td className="tabular-nums">{pct(e.metrics?.context_preservation)}</td>
                    <td className="tabular-nums">{pct(passHatK(e.metrics, e.k))}</td>
                    <td className="tabular-nums">{ms(e.metrics?.latency_ms_p95)}</td>
                    <td className="tabular-nums">{num(e.metrics?.throughput_tasks_per_min, 0)}<span className="text-[var(--muted)]">/min</span></td>
                    <td className="tabular-nums">{e.metrics?.n ?? e.n_tasks}</td>
                    <td className="whitespace-nowrap text-[var(--muted)]">{ts(e.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </DataState>
    </>
  );
}
