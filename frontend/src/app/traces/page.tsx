"use client";
import { Suspense, useState } from "react";
import useSWR from "swr";
import { fetcher, q, type Paged, type TraceSummary } from "@/lib/api";
import { ms, ts } from "@/lib/format";
import { Chip, DataState, IdLink, PageHeader } from "@/components/ui";

const PAGE = 25;

function TracesTable() {
  const [offset, setOffset] = useState(0);
  const [status, setStatus] = useState("");
  const { data, error } = useSWR<Paged<TraceSummary>>(q("/v1/traces", { limit: PAGE, offset, status }), fetcher, { refreshInterval: 5000, keepPreviousData: true });
  return (
    <>
      <PageHeader
        title="Traces"
        subtitle={data ? `${data.total.toLocaleString()} traces` : undefined}
        actions={
          <select className="input" value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}>
            <option value="">any status</option>
            <option value="ok">ok</option>
            <option value="error">error</option>
            <option value="unset">unset</option>
          </select>
        }
      />
      <DataState data={data} error={error} isEmpty={(d) => d.items.length === 0} empty="No traces ingested yet.">
        {(d) => (
          <>
            <div className="panel overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Trace</th>
                    <th>Name</th>
                    <th>Service</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Spans</th>
                    <th>Rollout</th>
                    <th>Input</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {d.items.map((t) => (
                    <tr key={t.id}>
                      <td><IdLink href={`/traces/detail?id=${t.id}`} id={t.id} /></td>
                      <td className="font-medium">{t.name}</td>
                      <td className="text-[var(--muted)]">{t.service}</td>
                      <td><Chip status={t.status} /></td>
                      <td className="tabular-nums">{ms(t.duration_ms)}</td>
                      <td className="tabular-nums">{t.n_spans}</td>
                      <td><IdLink href={`/rollouts/detail?id=${t.rollout_id}`} id={t.rollout_id} n={20} /></td>
                      <td className="max-w-[26rem] truncate text-[var(--muted)]" title={String(t.attributes?.["input.value"] ?? "")}>
                        {String(t.attributes?.["input.value"] ?? "")}
                      </td>
                      <td className="whitespace-nowrap text-[var(--muted)]">{ts(t.start_time)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between text-[12px] text-[var(--muted)]">
              <span>
                {offset + 1}–{Math.min(offset + PAGE, d.total)} of {d.total}
              </span>
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
      <TracesTable />
    </Suspense>
  );
}
