"use client";
import Link from "next/link";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { Api, fetcher, q, type Agent } from "@/lib/api";
import { ts } from "@/lib/format";
import { Chip, DataState, ErrorBox, IdLink, PageHeader } from "@/components/ui";

function AgentsTable() {
  const params = useSearchParams();
  const name = params.get("name") || undefined;
  const { data, error, mutate } = useSWR<Agent[]>(q("/v1/agents", { name }), fetcher, { refreshInterval: 5000 });
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  async function promote(id: string) {
    setBusy(id);
    setErr(null);
    try {
      await Api.promoteAgent(id);
      await mutate();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHeader
        title="Agents"
        subtitle={name ? <>Filtered to <span className="mono">{name}</span> · <Link href="/agents" className="link">clear</Link></> : "All agent versions in the project"}
      />
      {err ? <div className="mb-3"><ErrorBox error={err} /></div> : null}
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No agents registered.">
        {(agents) => (
          <div className="panel overflow-x-auto">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Version</th>
                  <th>Status</th>
                  <th>Origin</th>
                  <th>Model</th>
                  <th>Router top-k</th>
                  <th>Exemplar k</th>
                  <th>Parent</th>
                  <th>Created</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {agents.map((a) => (
                  <tr key={a.id}>
                    <td className="font-medium">
                      <Link href={`/agents/detail?id=${a.id}`} className="hover:underline">{a.name}</Link>
                      <div className="mono text-[11px] text-[var(--muted)]">{a.id}</div>
                    </td>
                    <td className="mono">{a.version}</td>
                    <td><Chip status={a.status} /></td>
                    <td>{a.origin}</td>
                    <td className="mono text-[12px]">{a.config?.model}</td>
                    <td className="tabular-nums">{a.config?.tool_router ? a.config.router_top_k ?? "–" : <span className="text-[var(--muted)]">no router</span>}</td>
                    <td className="tabular-nums">{a.config?.exemplar_store ? a.config.exemplar_k ?? "–" : <span className="text-[var(--muted)]">–</span>}</td>
                    <td><IdLink href={`/agents/detail?id=${a.parent_id}`} id={a.parent_id} /></td>
                    <td className="whitespace-nowrap text-[var(--muted)]">{ts(a.created_at)}</td>
                    <td className="whitespace-nowrap">
                      <Link href={`/agents/detail?id=${a.id}`} className="btn mr-2">Detail</Link>
                      <button className="btn btn-primary" disabled={a.status === "deployed" || busy === a.id} onClick={() => promote(a.id)}>
                        {busy === a.id ? "Promoting…" : a.status === "deployed" ? "Deployed" : "Promote"}
                      </button>
                    </td>
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

export default function AgentsPage() {
  return (
    <Suspense>
      <AgentsTable />
    </Suspense>
  );
}
