"use client";
import { useState } from "react";
import useSWR from "swr";
import { fetcher, type Environment, type ToolSpec } from "@/lib/api";
import { JsonView } from "@/components/JsonView";
import { DataState, Expander, PageHeader, Panel } from "@/components/ui";

function Tools({ name }: { name: string }) {
  const { data, error } = useSWR<ToolSpec[]>(`/v1/environments/${name}/tools`, fetcher);
  return (
    <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No tools.">
      {(tools) => (
        <table className="tbl">
          <thead><tr><th>Tool</th><th>Description</th><th>Parameters</th><th>Tags</th><th>Source</th></tr></thead>
          <tbody>
            {tools.map((t) => {
              const props = (t.parameters as { properties?: Record<string, { type?: string }>; required?: string[] }) || {};
              const req = new Set(props.required || []);
              return (
                <tr key={t.name}>
                  <td className="mono text-[12px] font-medium">{t.name}</td>
                  <td className="max-w-[28rem]">{t.description}</td>
                  <td className="mono text-[11px] text-[var(--muted)]">
                    {Object.entries(props.properties || {}).map(([k, v]) => (
                      <div key={k}>{k}{req.has(k) ? "" : "?"}: {v.type ?? "any"}</div>
                    ))}
                    {!Object.keys(props.properties || {}).length && "–"}
                  </td>
                  <td>
                    {t.tags?.map((g) => (
                      <span key={g} className={`mr-1 rounded px-1.5 text-[11px] ${g === "core" ? "bg-emerald-500/15 text-emerald-300" : "bg-zinc-500/15 text-zinc-300"}`}>{g}</span>
                    ))}
                  </td>
                  <td className="text-[var(--muted)]">{t.source}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </DataState>
  );
}

export default function EnvironmentsPage() {
  const { data, error } = useSWR<Environment[]>("/v1/environments", fetcher);
  const [open, setOpen] = useState<string | null>(null);
  return (
    <>
      <PageHeader title="Environments" subtitle="Tool environments agents are trained and evaluated in" />
      <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No environments registered.">
        {(envs) => (
          <div className="space-y-4">
            {envs.map((e) => (
              <Panel key={e.name} title={<span className="mono">{e.name}</span>} actions={<button className="btn" onClick={() => setOpen(open === e.name ? null : e.name)}>{open === e.name ? "Hide tool specs" : `Tool specs (${e.n_tools})`}</button>}>
                <div className="text-[13px]">{e.description}</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {e.tools.map((t) => <span key={t} className="mono rounded bg-[var(--panel-2)] px-1.5 py-0.5 text-[11px]">{t}</span>)}
                </div>
                {open === e.name && <div className="mt-4 overflow-x-auto"><Tools name={e.name} /></div>}
              </Panel>
            ))}
            <Expander title="Raw JSON"><JsonView value={envs} /></Expander>
          </div>
        )}
      </DataState>
    </>
  );
}
