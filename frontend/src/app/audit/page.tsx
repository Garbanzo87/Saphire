"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { fetcher, q, type AuditItem, type Paged } from "@/lib/api";
import { ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, Field, PageHeader } from "@/components/ui";

const PAGE = 50;
// resource_type -> console page that can show it
const RESOURCE_PAGES: Record<string, string> = {
  agent: "/agents/detail?id=",
  agents: "/agents/detail?id=",
  eval: "/evals/detail?id=",
  evals: "/evals/detail?id=",
  training: "/training/detail?id=",
  dataset: "/datasets?id=",
  datasets: "/datasets?id=",
  experiment: "/experiments?id=",
  experiments: "/experiments?id=",
  rollout: "/rollouts/detail?id=",
  trace: "/traces/detail?id=",
  job: "/jobs?id=",
};

function ResourceCell({ item }: { item: AuditItem }) {
  const base = RESOURCE_PAGES[item.resource_type];
  return (
    <div className="text-[12px]">
      <span>{item.resource_type || <span className="text-[var(--muted)]">–</span>}</span>
      {item.resource_id && (
        <div className="mono text-[11px]">
          {base ? <Link href={`${base}${encodeURIComponent(item.resource_id)}`} className="link">{item.resource_id}</Link> : <span className="text-[var(--muted)]">{item.resource_id}</span>}
        </div>
      )}
    </div>
  );
}

function Row({ item }: { item: AuditItem }) {
  const [open, setOpen] = useState(false);
  const hasDetails = item.details && Object.keys(item.details).length > 0;
  const family = `${Math.floor(item.status_code / 100)}xx`;
  return (
    <>
      <tr className={open ? "bg-[var(--panel-2)]" : ""}>
        <td className="whitespace-nowrap text-[var(--muted)]">{ts(item.created_at)}</td>
        <td>
          <div className="flex items-center gap-1.5">
            <span className="font-medium">{item.actor_label}</span>
            <Chip status={item.actor_type}>{item.actor_type}</Chip>
          </div>
        </td>
        <td className="mono text-[12px]">{item.action}</td>
        <td><ResourceCell item={item} /></td>
        <td className="mono text-[12px]">{item.method || <span className="text-[var(--muted)]">–</span>}<div className="text-[11px] text-[var(--muted)]">{item.path}</div></td>
        <td>{item.status_code ? <Chip status={family}>{item.status_code}</Chip> : <span className="text-[var(--muted)]" title="not an HTTP request (CLI / system)">–</span>}</td>
        <td className="mono text-[12px] text-[var(--muted)]">{item.ip || "–"}</td>
        <td className="text-right">
          <button type="button" className="btn" disabled={!hasDetails} onClick={() => setOpen(!open)} title={hasDetails ? "Show details" : "No details"}>
            {open ? "Hide" : "Details"}
          </button>
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={8} className="bg-[var(--panel-2)]">
            <JsonView value={item.details} />
          </td>
        </tr>
      )}
    </>
  );
}

export default function AuditPage() {
  const [action, setAction] = useState("");
  const [actor, setActor] = useState("");
  const [applied, setApplied] = useState({ action: "", actor: "" });
  const [offset, setOffset] = useState(0);
  const { data, error } = useSWR<Paged<AuditItem>>(q("/v1/audit", { limit: PAGE, offset, action: applied.action, actor: applied.actor }), fetcher, { keepPreviousData: true, refreshInterval: 10000 });

  function apply(e: React.FormEvent) {
    e.preventDefault();
    setApplied({ action: action.trim(), actor: actor.trim() });
    setOffset(0);
  }
  const total = data?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE, total);
  return (
    <>
      <PageHeader
        title="Audit log"
        subtitle={data ? `${total.toLocaleString()} events${applied.action || applied.actor ? " matching filters" : ""}` : undefined}
        actions={
          <form onSubmit={apply} className="flex flex-wrap items-end gap-2">
            <Field label="Action prefix">
              <input className="input w-44" value={action} onChange={(e) => setAction(e.target.value)} placeholder="e.g. agents." />
            </Field>
            <Field label="Actor">
              <input className="input w-44" value={actor} onChange={(e) => setActor(e.target.value)} placeholder="email, key name…" />
            </Field>
            <button className="btn btn-primary">Filter</button>
            {(applied.action || applied.actor) && (
              <button type="button" className="btn" onClick={() => { setAction(""); setActor(""); setApplied({ action: "", actor: "" }); setOffset(0); }}>Clear</button>
            )}
          </form>
        }
      />
      <DataState data={data} error={error} isEmpty={(d) => d.items.length === 0} empty="No audit events match.">
        {(page) => (
          <div className="space-y-3">
            <div className="panel overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr><th>Time</th><th>Actor</th><th>Action</th><th>Resource</th><th>Request</th><th>Status</th><th>IP</th><th></th></tr>
                </thead>
                <tbody>
                  {page.items.map((it) => <Row key={it.id} item={it} />)}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between text-[12px] text-[var(--muted)]">
              <span>
                Showing {from}–{to} of {total.toLocaleString()}
              </span>
              <div className="flex gap-2">
                <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← Newer</button>
                <button className="btn" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Older →</button>
              </div>
            </div>
          </div>
        )}
      </DataState>
    </>
  );
}
