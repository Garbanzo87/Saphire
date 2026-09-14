"use client";
import Link from "next/link";
import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { fetcher, type Span, type TraceDetail } from "@/lib/api";
import { ms, num, ts, tryJson } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, ErrorBox, PageHeader, Panel, Stat } from "@/components/ui";

const kindColor: Record<string, string> = { agent: "bg-violet-400", llm: "bg-sky-400", tool: "bg-amber-400" };

function orderSpans(spans: Span[]) {
  // depth-first ordering by parent, siblings sorted by start time
  const byParent = new Map<string | null, Span[]>();
  const ids = new Set(spans.map((s) => s.id));
  for (const s of spans) {
    const p = s.parent_span_id && ids.has(s.parent_span_id) ? s.parent_span_id : null;
    byParent.set(p, [...(byParent.get(p) || []), s]);
  }
  const out: { span: Span; depth: number }[] = [];
  const visit = (parent: string | null, depth: number) => {
    for (const s of (byParent.get(parent) || []).sort((a, b) => a.start_time - b.start_time)) {
      out.push({ span: s, depth });
      visit(s.id, depth + 1);
    }
  };
  visit(null, 0);
  return out;
}

function TraceView() {
  const id = useSearchParams().get("id") || "";
  const { data, error } = useSWR<TraceDetail>(id ? `/v1/traces/${id}` : null, fetcher);
  const [selected, setSelected] = useState<string | null>(null);
  const rows = useMemo(() => (data ? orderSpans(data.spans) : []), [data]);
  if (!id) return <ErrorBox error="Missing ?id=" />;
  return (
    <DataState data={data} error={error}>
      {(t) => {
        const t0 = t.start_time;
        const total = Math.max(1e-6, t.end_time - t.start_time);
        const sel = t.spans.find((s) => s.id === selected) || rows[0]?.span;
        const attrs = sel?.attributes || {};
        const { "input.value": input, "output.value": output, ...otherAttrs } = attrs as Record<string, unknown>;
        return (
          <>
            <PageHeader
              title={t.name}
              subtitle={
                <span className="flex flex-wrap items-center gap-2">
                  <span className="mono">{t.id}</span> · <Chip status={t.status} /> · {t.service} · {ts(t.start_time)}
                  {t.rollout_id && (
                    <>· rollout <Link href={`/rollouts/detail?id=${t.rollout_id}`} className="link mono">{t.rollout_id}</Link></>
                  )}
                </span>
              }
              actions={<Link href="/traces" className="btn">← Traces</Link>}
            />
            <div className="mb-4 grid grid-cols-2 gap-2 md:grid-cols-5">
              <Stat label="Duration" value={ms(total * 1000)} />
              <Stat label="Spans" value={t.n_spans} />
              <Stat label="LLM calls" value={t.spans.filter((s) => s.kind === "llm").length} />
              <Stat label="Tool calls" value={t.spans.filter((s) => s.kind === "tool").length} />
              <Stat label="Scores" value={t.scores.length} />
            </div>
            <div className="grid gap-4 xl:grid-cols-5">
              <Panel title="Spans" className="xl:col-span-3">
                <div className="space-y-0.5">
                  {rows.map(({ span, depth }) => {
                    const left = ((span.start_time - t0) / total) * 100;
                    const width = Math.max(0.5, ((span.end_time - span.start_time) / total) * 100);
                    const active = sel?.id === span.id;
                    return (
                      <button
                        key={span.id}
                        type="button"
                        onClick={() => setSelected(span.id)}
                        className={`grid w-full grid-cols-[minmax(0,17rem)_1fr_5rem] items-center gap-2 rounded px-1.5 py-1 text-left text-[12px] ${active ? "bg-[var(--panel-2)]" : "hover:bg-[var(--panel-2)]"}`}
                      >
                        <span className="flex min-w-0 items-center gap-1.5" style={{ paddingLeft: depth * 14 }}>
                          <Chip status={span.kind}>{span.kind}</Chip>
                          <span className="truncate" title={span.name}>{span.name}</span>
                        </span>
                        <span className="relative h-3.5 rounded bg-[var(--bg)]">
                          <span className={`absolute top-0 h-full rounded ${kindColor[span.kind] || "bg-zinc-400"} ${span.status === "error" ? "ring-1 ring-rose-400" : ""}`} style={{ left: `${left}%`, width: `${width}%` }} />
                        </span>
                        <span className="text-right tabular-nums text-[var(--muted)]">{ms((span.end_time - span.start_time) * 1000)}</span>
                      </button>
                    );
                  })}
                </div>
              </Panel>
              <div className="space-y-4 xl:col-span-2">
                <Panel title={sel ? <span className="flex items-center gap-2"><Chip status={sel.kind}>{sel.kind}</Chip> {sel.name}</span> : "Span"}>
                  {sel ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-2 text-[12px]">
                        <div><span className="label">start</span><div className="tabular-nums">+{num((sel.start_time - t0) * 1000, 2)} ms</div></div>
                        <div><span className="label">duration</span><div className="tabular-nums">{ms((sel.end_time - sel.start_time) * 1000)}</div></div>
                        <div><span className="label">status</span><div><Chip status={sel.status} /></div></div>
                        <div><span className="label">span id</span><div className="mono">{sel.id}</div></div>
                      </div>
                      {Object.keys(otherAttrs).length > 0 && (
                        <div>
                          <div className="label mb-1">attributes</div>
                          <JsonView value={Object.fromEntries(Object.entries(otherAttrs).map(([k, v]) => [k, tryJson(v)]))} />
                        </div>
                      )}
                      {input !== undefined && (
                        <div>
                          <div className="label mb-1">input.value</div>
                          <ValueView v={input} />
                        </div>
                      )}
                      {output !== undefined && (
                        <div>
                          <div className="label mb-1">output.value</div>
                          <ValueView v={output} />
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-[var(--muted)]">No spans.</div>
                  )}
                </Panel>
                <Panel title="Scores">
                  {t.scores.length === 0 ? (
                    <div className="text-[var(--muted)]">No scores attached.</div>
                  ) : (
                    <table className="tbl">
                      <thead>
                        <tr><th>Name</th><th>Value</th><th>Source</th><th>Step</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {t.scores.map((s) => (
                          <tr key={s.id}>
                            <td className="mono text-[12px]">{s.name}</td>
                            <td className={`tabular-nums ${s.value >= 1 ? "text-emerald-300" : s.value <= 0 ? "text-rose-300" : ""}`}>{num(s.value, 2)}</td>
                            <td>{s.source}</td>
                            <td className="tabular-nums">{s.step_index ?? "–"}</td>
                            <td className="text-[12px] text-[var(--muted)]">{s.rationale || "–"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </Panel>
              </div>
            </div>
          </>
        );
      }}
    </DataState>
  );
}

function ValueView({ v }: { v: unknown }) {
  const parsed = tryJson(v);
  if (typeof parsed === "string") return <pre className="mono max-h-72 overflow-auto whitespace-pre-wrap rounded-md bg-[var(--bg)] p-3 text-[12px] leading-5">{parsed}</pre>;
  return <JsonView value={parsed} />;
}

export default function Page() {
  return (
    <Suspense>
      <TraceView />
    </Suspense>
  );
}
