"use client";
import { useState } from "react";
import useSWR from "swr";
import { fetcher, q, type Dataset } from "@/lib/api";
import { ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, Expander, PageHeader, Panel } from "@/components/ui";

function TaskPreview({ id }: { id: string }) {
  const { data, error } = useSWR<Dataset>(q(`/v1/datasets/${id}`, { limit: 8 }), fetcher);
  return (
    <DataState data={data} error={error} isEmpty={(d) => !d.tasks?.length} empty="No tasks.">
      {(d) => (
        <div className="space-y-2">
          {(d.tasks || []).map((t) => (
            <div key={t.id} className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-3 text-[13px]">
              <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-[var(--muted)]">
                <span className="mono">{t.id}</span> · {t.env_name} · <Chip status={t.difficulty}>{t.difficulty}</Chip> · max {t.max_steps} steps
                {t.tags?.map((g) => <span key={g} className="rounded bg-[var(--panel-2)] px-1.5">{g}</span>)}
              </div>
              <div>{t.instruction}</div>
              {t.user_script?.length > 0 && (
                <ol className="mt-1 list-decimal pl-5 text-[12px] text-[var(--muted)]">
                  {t.user_script.map((u, i) => <li key={i}>{u}</li>)}
                </ol>
              )}
              <div className="mt-1 text-[11px] text-[var(--muted)]">expects tools: <span className="mono">{((t.expected?.tools as string[]) || []).join(", ") || "–"}</span></div>
            </div>
          ))}
          {d.n_tasks > (d.tasks?.length || 0) && <div className="text-[12px] text-[var(--muted)]">showing {d.tasks?.length} of {d.n_tasks} tasks</div>}
        </div>
      )}
    </DataState>
  );
}

export default function DatasetsPage() {
  const { data, error } = useSWR<Dataset[]>("/v1/datasets", fetcher, { refreshInterval: 5000 });
  const [open, setOpen] = useState<string | null>(null);
  return (
    <>
      <PageHeader title="Datasets" subtitle="Task collections used for training and evaluation" />
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No datasets.">
        {(rows) => (
          <div className="space-y-4">
            <Panel>
              <table className="tbl">
                <thead><tr><th>Name</th><th>ID</th><th>Suite</th><th>Split</th><th>Tasks</th><th>Created</th><th></th></tr></thead>
                <tbody>
                  {rows.map((d) => (
                    <tr key={d.id}>
                      <td className="font-medium">{d.name}</td>
                      <td className="mono text-[12px]">{d.id}</td>
                      <td>{d.suite || "–"}</td>
                      <td><Chip status={d.split === "train" ? "candidate" : "succeeded"}>{d.split}</Chip></td>
                      <td className="tabular-nums">{d.n_tasks}</td>
                      <td className="whitespace-nowrap text-[var(--muted)]">{ts(d.created_at)}</td>
                      <td><button className="btn" onClick={() => setOpen(open === d.id ? null : d.id)}>{open === d.id ? "Hide tasks" : "Preview tasks"}</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
            {open && (
              <Panel title={`Tasks · ${rows.find((r) => r.id === open)?.name}`}>
                <TaskPreview id={open} />
              </Panel>
            )}
            <Expander title="Raw JSON"><JsonView value={rows} /></Expander>
          </div>
        )}
      </DataState>
    </>
  );
}
