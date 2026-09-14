"use client";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { fetcher, q, type Agent, type Environment, type Paged, type Rollout } from "@/lib/api";
import { ms, pct, ts } from "@/lib/format";
import { Chip, DataState, IdLink, PageHeader } from "@/components/ui";

const PAGE = 25;

function RolloutsTable() {
  const sp = useSearchParams();
  const [offset, setOffset] = useState(0);
  const [env, setEnv] = useState("");
  const [success, setSuccess] = useState("");
  const [agent, setAgent] = useState("");
  const evalRun = sp.get("eval_run_id") || undefined;
  const trainingRun = sp.get("training_run_id") || undefined;
  const { data, error } = useSWR<Paged<Rollout>>(
    q("/v1/rollouts", { limit: PAGE, offset, env, success, agent_id: agent, eval_run_id: evalRun, training_run_id: trainingRun }),
    fetcher,
    { refreshInterval: 5000, keepPreviousData: true },
  );
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const { data: envs } = useSWR<Environment[]>("/v1/environments", fetcher);
  const reset = () => setOffset(0);
  return (
    <>
      <PageHeader
        title="Rollouts"
        subtitle={data ? `${data.total.toLocaleString()} rollouts${evalRun ? ` · eval ${evalRun}` : ""}${trainingRun ? ` · training ${trainingRun}` : ""}` : undefined}
        actions={
          <>
            <select className="input" value={env} onChange={(e) => { setEnv(e.target.value); reset(); }}>
              <option value="">any env</option>
              {(envs || []).map((e) => <option key={e.name} value={e.name}>{e.name}</option>)}
            </select>
            <select className="input" value={success} onChange={(e) => { setSuccess(e.target.value); reset(); }}>
              <option value="">success + fail</option>
              <option value="true">success only</option>
              <option value="false">fail only</option>
            </select>
            <select className="input" value={agent} onChange={(e) => { setAgent(e.target.value); reset(); }}>
              <option value="">any agent</option>
              {(agents || []).map((a) => <option key={a.id} value={a.id}>{a.name}/{a.version}</option>)}
            </select>
          </>
        }
      />
      <DataState data={data} error={error} isEmpty={(d) => d.items.length === 0} empty="No rollouts match the filter.">
        {(d) => (
          <>
            <div className="panel overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Rollout</th>
                    <th>Env</th>
                    <th>Family</th>
                    <th>Difficulty</th>
                    <th>Status</th>
                    <th>Task success</th>
                    <th>Tool F1</th>
                    <th>Steps</th>
                    <th>Tool calls</th>
                    <th>Latency</th>
                    <th>Trace</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {d.items.map((r) => (
                    <tr key={r.id}>
                      <td><IdLink href={`/rollouts/detail?id=${r.id}`} id={r.id} n={22} /></td>
                      <td>{r.env_name}</td>
                      <td>{r.record?.family ?? "–"}</td>
                      <td>{r.record?.difficulty ?? "–"}</td>
                      <td><Chip status={r.status} /></td>
                      <td className={`tabular-nums ${(r.task_success ?? 0) >= 1 ? "text-emerald-300" : "text-rose-300"}`}>{pct(r.task_success)}</td>
                      <td className="tabular-nums">{pct(r.record?.tool_selection_f1)}</td>
                      <td className="tabular-nums">{r.record?.n_steps ?? "–"}</td>
                      <td className="tabular-nums">{r.record?.n_tool_calls ?? "–"}</td>
                      <td className="tabular-nums">{ms(r.record?.latency_ms)}</td>
                      <td><IdLink href={`/traces/detail?id=${r.trace_id}`} id={r.trace_id} n={10} /></td>
                      <td className="whitespace-nowrap text-[var(--muted)]">{ts(r.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between text-[12px] text-[var(--muted)]">
              <span>{offset + 1}–{Math.min(offset + PAGE, d.total)} of {d.total}</span>
              <div className="flex gap-2">
                <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Prev</button>
                <button className="btn" disabled={offset + PAGE >= d.total} onClick={() => setOffset(offset + PAGE)}>Next →</button>
              </div>
            </div>
          </>
        )}
      </DataState>
    </>
  );
}

export default function Page() {
  return (
    <Suspense>
      <RolloutsTable />
    </Suspense>
  );
}
