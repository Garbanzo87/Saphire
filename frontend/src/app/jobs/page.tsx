"use client";
import Link from "next/link";
import { Fragment, Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { Api, fetcher, q, type Job } from "@/lib/api";
import { isActive, ms, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, ErrorBox, Logs, PageHeader, Progress } from "@/components/ui";

function targetLink(j: Job) {
  const p = j.params as Record<string, string | undefined>;
  if (p.eval_run_id) return <Link href={`/evals/detail?id=${p.eval_run_id}`} className="link mono text-[12px]">{p.eval_run_id}</Link>;
  if (p.training_run_id) return <Link href={`/training/detail?id=${p.training_run_id}`} className="link mono text-[12px]">{p.training_run_id}</Link>;
  return <span className="text-[var(--muted)]">–</span>;
}

function JobsTable() {
  const initial = useSearchParams().get("id");
  const [status, setStatus] = useState("");
  const [open, setOpen] = useState<string | null>(initial);
  const { data, error, mutate } = useSWR<Job[]>(q("/v1/jobs", { status, limit: 100 }), fetcher, { refreshInterval: 5000 });
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  return (
    <>
      <PageHeader
        title="Jobs"
        subtitle={data ? `${data.length} jobs · ${data.filter((j) => isActive(j.status)).length} active` : undefined}
        actions={
          <select className="input" value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">any status</option>
            {["pending", "running", "succeeded", "failed", "cancelled"].map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        }
      />
      {err ? <div className="mb-3"><ErrorBox error={err} /></div> : null}
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No jobs.">
        {(rows) => (
          <div className="panel overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Job</th><th>Type</th><th>Status</th><th>Progress</th><th>Target</th><th>Worker</th><th>Duration</th><th>Created</th><th>Error</th><th></th></tr></thead>
              <tbody>
                {rows.map((j) => {
                  const expanded = open === j.id;
                  const dur = j.started_at ? ((j.finished_at ?? Date.now() / 1000) - j.started_at) * 1000 : null;
                  return (
                    <Fragment key={j.id}>
                      <tr className="cursor-pointer" onClick={() => setOpen(expanded ? null : j.id)}>
                        <td className="mono text-[12px]"><span className="mr-1 text-[var(--muted)]">{expanded ? "▾" : "▸"}</span>{j.id}</td>
                        <td>{j.type}</td>
                        <td><Chip status={j.status} /></td>
                        <td><Progress value={j.progress} /></td>
                        <td>{targetLink(j)}</td>
                        <td className="mono text-[11px] text-[var(--muted)]">{j.worker || "–"}</td>
                        <td className="tabular-nums">{ms(dur)}</td>
                        <td className="whitespace-nowrap text-[var(--muted)]">{ts(j.created_at)}</td>
                        <td className="max-w-[20rem] truncate text-rose-300" title={j.error}>{j.error || ""}</td>
                        <td onClick={(e) => e.stopPropagation()}>
                          {isActive(j.status) && (
                            <button className="btn" disabled={busy === j.id} onClick={async () => { setBusy(j.id); setErr(null); try { await Api.cancelJob(j.id); mutate(); } catch (x) { setErr(x); } finally { setBusy(null); } }}>Cancel</button>
                          )}
                        </td>
                      </tr>
                      {expanded && (
                        <tr>
                          <td colSpan={10} className="bg-[var(--bg)]">
                            <div className="grid gap-3 lg:grid-cols-2">
                              <div>
                                <div className="label mb-1">logs</div>
                                <Logs logs={j.logs} />
                              </div>
                              <div className="space-y-3">
                                <div>
                                  <div className="label mb-1">params</div>
                                  <JsonView value={j.params} className="max-h-40" />
                                </div>
                                <div>
                                  <div className="label mb-1">result</div>
                                  <JsonView value={j.result} className="max-h-60" />
                                </div>
                              </div>
                            </div>
                          </td>
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
    </>
  );
}

export default function Page() {
  return (
    <Suspense>
      <JobsTable />
    </Suspense>
  );
}
