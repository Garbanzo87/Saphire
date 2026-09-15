"use client";
import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";
import { Api, fetcher, q, type Agent, type Dataset, type TrainingRun } from "@/lib/api";
import { delta, isNum, ts } from "@/lib/format";
import { Chip, DataState, ErrorBox, Field, IdLink, PageHeader, Panel } from "@/components/ui";

const ALGOS = ["online", "router", "exemplars", "prompt_opt", "signals", "sft", "dpo", "grpo"] as const;
type Algo = (typeof ALGOS)[number];
const DESC: Record<Algo, string> = {
  online: "Iterative loop: collect rollouts, learn router + exemplars (+ prompt every N iters), evaluate each iteration.",
  router: "Learn a tool router from stored rollouts / traces.",
  exemplars: "Build an exemplar store from successful rollouts.",
  prompt_opt: "Optimise the system prompt against a train dataset.",
  signals: "Derive training signals from traces and scores.",
  sft: "Supervised fine-tune (LoRA) on successful trajectories.",
  dpo: "Direct preference optimisation on paired trajectories.",
  grpo: "Group-relative policy optimisation with verifier rewards.",
};

function improvement(t: TrainingRun) {
  const imp = (t.result as { improvement?: Record<string, number> } | null)?.improvement;
  return imp?.task_success;
}

function NewTrainingForm({ onCreated }: { onCreated: (t: TrainingRun) => void }) {
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const { data: datasets } = useSWR<Dataset[]>("/v1/datasets", fetcher);
  const [algo, setAlgo] = useState<Algo>("online");
  const [agentId, setAgentId] = useState("");
  const [p, setP] = useState<Record<string, string | number>>({ iterations: 3, batch_size: 24, learn_prompt_every: 2, model: "mock", max_steps: 100, lora_r: 8, rollout_backend: "", rollout_workers: 4 });
  const [optimizeRoles, setOptimizeRoles] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const train = datasets?.filter((d) => d.split === "train") || [];
  const evals = datasets?.filter((d) => d.split !== "train") || [];
  const aid = agentId || agents?.find((a) => a.status === "deployed")?.id || agents?.[0]?.id || "";
  const selectedAgent = agents?.find((a) => a.id === aid);
  const roleNames = Object.keys(selectedAgent?.config?.roles || {});
  const multiAgent = roleNames.length > 0;
  const rolesSupported = algo === "online" || algo === "router" || algo === "exemplars";
  const trainId = String(p.train_dataset_id || train[0]?.id || datasets?.[0]?.id || "");
  const evalId = String(p.eval_dataset_id || evals[0]?.id || datasets?.[0]?.id || "");
  const set = (k: string, v: string | number) => setP({ ...p, [k]: v });
  const numIn = (k: string, label: string) => (
    <Field label={label}>
      <input className="input" type="number" value={p[k] ?? ""} onChange={(e) => set(k, Number(e.target.value))} />
    </Field>
  );
  const dsIn = (k: string, label: string, opts: Dataset[], val: string) => (
    <Field label={label}>
      <select className="input" value={val} onChange={(e) => set(k, e.target.value)}>
        {(opts.length ? opts : datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name} · {d.split} · {d.n_tasks}</option>)}
      </select>
    </Field>
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    let params: Record<string, unknown> = {};
    if (algo === "online") {
      params = { train_dataset_id: trainId, eval_dataset_id: evalId, iterations: p.iterations, batch_size: p.batch_size, learn_prompt_every: p.learn_prompt_every };
      if (p.rollout_backend) params = { ...params, rollout_backend: p.rollout_backend, rollout_workers: Number(p.rollout_workers) || undefined };
    }
    if (multiAgent && rolesSupported && optimizeRoles.length > 0) params = { ...params, optimize_roles: optimizeRoles.filter((r) => roleNames.includes(r)) };
    else if (algo === "prompt_opt") params = { train_dataset_id: trainId, iterations: p.iterations };
    else if (algo === "sft" || algo === "dpo" || algo === "grpo") params = { model: p.model, max_steps: p.max_steps, lora_r: p.lora_r };
    try {
      onCreated(await Api.createTraining({ agent_id: aid, algorithm: algo, params }));
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="grid gap-3 md:grid-cols-4">
        <Field label="Agent (input version)">
          <select className="input" value={aid} onChange={(e) => { setAgentId(e.target.value); setOptimizeRoles([]); }}>
            {(agents || []).map((a) => <option key={a.id} value={a.id}>{a.name}/{a.version} ({a.status}){a.config?.roles && Object.keys(a.config.roles).length ? ` · ${Object.keys(a.config.roles).length} roles` : ""}</option>)}
          </select>
        </Field>
        <Field label="Algorithm">
          <select className="input" value={algo} onChange={(e) => setAlgo(e.target.value as Algo)}>
            {ALGOS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </Field>
        <div className="md:col-span-2 self-end text-[12px] text-[var(--muted)]">{DESC[algo]}</div>
      </div>
      <div className="grid gap-3 md:grid-cols-5">
        {algo === "online" && (
          <>
            {dsIn("train_dataset_id", "Train dataset", train, trainId)}
            {dsIn("eval_dataset_id", "Eval dataset", evals, evalId)}
            {numIn("iterations", "Iterations")}
            {numIn("batch_size", "Batch size")}
            {numIn("learn_prompt_every", "Learn prompt every")}
            <Field label="Rollout backend">
              <select className="input" value={String(p.rollout_backend ?? "")} onChange={(e) => set("rollout_backend", e.target.value)}>
                <option value="">none (in-process)</option>
                <option value="thread">thread</option>
                <option value="process">process</option>
                <option value="ray">ray</option>
              </select>
            </Field>
            {p.rollout_backend ? numIn("rollout_workers", "Rollout workers") : null}
          </>
        )}
        {algo === "prompt_opt" && (
          <>
            {dsIn("train_dataset_id", "Train dataset", train, trainId)}
            {numIn("iterations", "Iterations")}
          </>
        )}
        {(algo === "sft" || algo === "dpo" || algo === "grpo") && (
          <>
            <Field label="Base model">
              <input className="input" value={String(p.model ?? "")} onChange={(e) => set("model", e.target.value)} />
            </Field>
            {numIn("max_steps", "Max steps")}
            {numIn("lora_r", "LoRA r")}
          </>
        )}
        {(algo === "router" || algo === "exemplars" || algo === "signals") && <div className="self-end text-[12px] text-[var(--muted)] md:col-span-3">No parameters — uses rollouts and traces already stored for this agent.</div>}
        {multiAgent && rolesSupported && (
          <Field label="Optimise roles (empty = all trainable)">
            <div className="flex flex-wrap gap-1.5 py-1" role="group" aria-label="Optimise roles">
              {roleNames.map((r) => {
                const on = optimizeRoles.includes(r);
                const trainable = selectedAgent?.config?.roles?.[r]?.trainable !== false;
                return (
                  <button
                    key={r}
                    type="button"
                    title={trainable ? "" : "role is marked non-trainable"}
                    onClick={() => setOptimizeRoles(on ? optimizeRoles.filter((x) => x !== r) : [...optimizeRoles, r])}
                    className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${on ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300" : "border-[var(--border)] bg-[var(--bg)] text-[var(--muted)] hover:text-white"} ${trainable ? "" : "line-through"}`}
                  >
                    {r}
                  </button>
                );
              })}
            </div>
          </Field>
        )}
        <div className="flex items-end">
          <button className="btn btn-primary w-full" disabled={busy || !aid}>{busy ? "Starting…" : "Start training"}</button>
        </div>
      </div>
      {err ? <ErrorBox error={err} /> : null}
    </form>
  );
}

export default function TrainingPage() {
  const router = useRouter();
  const { data, error, mutate } = useSWR<TrainingRun[]>(q("/v1/training", { limit: 100 }), fetcher, { refreshInterval: 5000 });
  const [showForm, setShowForm] = useState(false);
  return (
    <>
      <PageHeader title="Training" subtitle={data ? `${data.length} training runs` : undefined} actions={<button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "New training run"}</button>} />
      {showForm && (
        <Panel title="New training run" className="mb-4">
          <NewTrainingForm onCreated={(t) => { mutate(); router.push(`/training/detail?id=${t.id}`); }} />
        </Panel>
      )}
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No training runs yet.">
        {(rows) => (
          <div className="panel overflow-x-auto">
            <table className="tbl">
              <thead>
                <tr><th>Run</th><th>Agent</th><th>Algorithm</th><th>Input → Output</th><th>Status</th><th>Iterations</th><th>Task success Δ</th><th>Output agent</th><th>Created</th><th>Finished</th></tr>
              </thead>
              <tbody>
                {rows.map((t) => {
                  const imp = improvement(t);
                  return (
                    <tr key={t.id}>
                      <td><IdLink href={`/training/detail?id=${t.id}`} id={t.id} n={22} /></td>
                      <td><Link href={`/agents/detail?id=${t.agent_id}`} className="hover:underline">{t.agent_name ?? t.agent_id}</Link></td>
                      <td className="mono">{t.algorithm}</td>
                      <td className="mono">{t.input_version ?? "?"} → {t.output_version ?? <span className="text-[var(--muted)]">–</span>}</td>
                      <td><Chip status={t.status} /></td>
                      <td className="tabular-nums">{t.history?.length ? t.history.length - 1 : (t.params?.iterations as number) ?? "–"}</td>
                      <td className={`tabular-nums ${isNum(imp) ? (imp > 0 ? "text-emerald-300" : imp < 0 ? "text-rose-300" : "") : "text-[var(--muted)]"}`}>{delta(imp)}</td>
                      <td><IdLink href={`/agents/detail?id=${t.output_agent_id}`} id={t.output_agent_id} n={20} /></td>
                      <td className="whitespace-nowrap text-[var(--muted)]">{ts(t.created_at)}</td>
                      <td className="whitespace-nowrap text-[var(--muted)]">{ts(t.finished_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </DataState>
    </>
  );
}
