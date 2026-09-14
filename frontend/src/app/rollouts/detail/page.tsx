"use client";
import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { fetcher, type ChatMessage, type RolloutDetail } from "@/lib/api";
import { ms, num, pct, ts, tryJson } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, ErrorBox, Expander, IdLink, PageHeader, Panel, Stat } from "@/components/ui";

const roleStyle: Record<string, string> = {
  system: "border-zinc-500/40 text-zinc-300",
  user: "border-sky-500/40",
  assistant: "border-violet-500/40",
  tool: "border-amber-500/40",
};

function Msg({ m }: { m: ChatMessage }) {
  const parsed = m.role === "tool" ? tryJson(m.content) : m.content;
  return (
    <div className={`rounded-md border-l-2 bg-[var(--bg)] px-3 py-2 ${roleStyle[m.role] || "border-zinc-500/40"}`}>
      <div className="label mb-1">
        {m.role}
        {m.name ? <span className="mono normal-case"> · {m.name}</span> : null}
      </div>
      {m.content && (typeof parsed === "string" ? <div className="whitespace-pre-wrap text-[13px]">{parsed}</div> : <JsonView value={parsed} className="max-h-60" />)}
      {m.tool_calls?.length > 0 && (
        <div className="mt-1 space-y-1">
          {m.tool_calls.map((c) => (
            <div key={c.id} className="mono text-[12px] text-amber-200">
              {c.name}({JSON.stringify(c.arguments)})
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RolloutView() {
  const id = useSearchParams().get("id") || "";
  const { data, error } = useSWR<RolloutDetail>(id ? `/v1/rollouts/${id}` : null, fetcher);
  if (!id) return <ErrorBox error="Missing ?id=" />;
  return (
    <DataState data={data} error={error}>
      {(r) => {
        const rec = r.record;
        const p = r.payload;
        const steps = p?.steps || [];
        const rewards = p?.rewards || [];
        const taskReward = rewards.find((x) => x.name === "task_success");
        const checks = (taskReward?.metadata?.checks as Record<string, boolean> | undefined) || undefined;
        return (
          <>
            <PageHeader
              title={`Rollout ${r.id}`}
              subtitle={
                <span className="flex flex-wrap items-center gap-2">
                  <Chip status={r.status} /> {r.env_name} · {rec?.family} / {rec?.difficulty} · task <span className="mono">{r.task_id}</span> · {ts(r.created_at)}
                </span>
              }
              actions={
                <>
                  {r.trace_id && <Link href={`/traces/detail?id=${r.trace_id}`} className="btn">Trace</Link>}
                  {r.eval_run_id && <Link href={`/evals/detail?id=${r.eval_run_id}`} className="btn">Eval run</Link>}
                  {r.training_run_id && <Link href={`/training/detail?id=${r.training_run_id}`} className="btn">Training run</Link>}
                  <Link href={`/agents/detail?id=${r.agent_id}`} className="btn">Agent {p?.agent_version ?? ""}</Link>
                </>
              }
            />
            <div className="mb-4 grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
              <Stat label="Task success" value={<span className={(r.task_success ?? 0) >= 1 ? "text-emerald-300" : "text-rose-300"}>{pct(r.task_success)}</span>} />
              <Stat label="Total reward" value={num(r.total_reward, 3)} />
              <Stat label="Tool F1" value={pct(rec?.tool_selection_f1)} />
              <Stat label="Step efficiency" value={pct(rec?.step_efficiency)} />
              <Stat label="Context" value={pct(rec?.context_preservation)} />
              <Stat label="Steps" value={rec?.n_steps ?? "–"} />
              <Stat label="Tool calls" value={rec?.n_tool_calls ?? "–"} />
              <Stat label="Latency" value={ms(rec?.latency_ms)} />
            </div>
            <div className="grid gap-4 xl:grid-cols-5">
              <div className="space-y-4 xl:col-span-3">
                <Panel title={`Conversation · ${steps.length} step${steps.length === 1 ? "" : "s"}`}>
                  {steps.length === 0 ? (
                    <div className="text-[var(--muted)]">No step payload stored for this rollout.</div>
                  ) : (
                    <div className="space-y-5">
                      {steps.map((s, i) => {
                        // Show only messages that are new relative to the previous step's prompt.
                        const prevLen = i === 0 ? 0 : steps[i - 1].prompt_messages.length + 1 + steps[i - 1].tool_results.length;
                        const fresh = s.prompt_messages.slice(i === 0 ? 0 : Math.min(prevLen, s.prompt_messages.length));
                        const hidden = s.prompt_messages.length - fresh.length;
                        return (
                          <div key={s.index}>
                            <div className="mb-2 flex items-center justify-between text-[12px] text-[var(--muted)]">
                              <span className="font-medium text-[var(--fg)]">Step {s.index}</span>
                              <span className="tabular-nums">
                                {s.usage?.prompt_tokens ?? "–"} → {s.usage?.completion_tokens ?? "–"} tok · {ms(s.latency_ms)} · {s.exposed_tools?.length ?? 0} tools exposed
                                {s.reward !== null && s.reward !== undefined ? ` · reward ${num(s.reward, 2)}` : ""}
                              </span>
                            </div>
                            <div className="space-y-1.5">
                              {hidden > 0 && <div className="text-[11px] text-[var(--muted)]">… {hidden} earlier message{hidden === 1 ? "" : "s"} carried over</div>}
                              {fresh.map((m, j) => <Msg key={j} m={m} />)}
                              <Msg m={s.response} />
                              {s.tool_results.map((tr) => (
                                <div key={tr.call_id} className={`rounded-md border-l-2 bg-[var(--bg)] px-3 py-2 ${tr.error ? "border-rose-500/60" : "border-amber-500/40"}`}>
                                  <div className="label mb-1">
                                    tool result · <span className="mono normal-case">{tr.name}</span> · {ms(tr.latency_ms)}
                                  </div>
                                  {tr.error ? <div className="text-[13px] text-rose-300">{tr.error}</div> : typeof tr.output === "string" ? <div className="whitespace-pre-wrap text-[13px]">{tr.output}</div> : <JsonView value={tr.output} className="max-h-60" />}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                      {p?.final_answer && (
                        <div>
                          <div className="label mb-1">final answer</div>
                          <div className="whitespace-pre-wrap rounded-md border-l-2 border-emerald-500/50 bg-[var(--bg)] px-3 py-2 text-[13px]">{p.final_answer}</div>
                        </div>
                      )}
                    </div>
                  )}
                </Panel>
              </div>
              <div className="space-y-4 xl:col-span-2">
                <Panel title="Verifier checks">
                  {checks ? (
                    <ul className="space-y-1">
                      {Object.entries(checks).map(([k, ok]) => (
                        <li key={k} className="flex items-center gap-2 text-[13px]">
                          <Chip status={ok}>{ok ? "ok" : "fail"}</Chip>
                          <span className="mono text-[12px]">{k}</span>
                        </li>
                      ))}
                    </ul>
                  ) : rec?.rationale ? (
                    <div className="mono text-[12px]">{rec.rationale}</div>
                  ) : (
                    <div className="text-[var(--muted)]">No verifier checks recorded.</div>
                  )}
                  {rec?.called_tools?.length ? (
                    <div className="mt-3 text-[12px]">
                      <span className="label">called tools</span>
                      <div className="mono">{rec.called_tools.join(" → ")}</div>
                    </div>
                  ) : null}
                </Panel>
                <Panel title="Rewards">
                  {rewards.length === 0 ? (
                    <div className="text-[var(--muted)]">No rewards.</div>
                  ) : (
                    <table className="tbl table-fixed">
                      <thead><tr><th className="w-auto">Name</th><th className="w-16">Value</th><th className="w-20">Source</th><th className="w-14">Step</th></tr></thead>
                      <tbody>
                        {rewards.map((w, i) => (
                          <tr key={i}>
                            <td>
                              <div className="mono text-[12px]">{w.name}</div>
                              {w.rationale && <div className="break-words text-[11px] text-[var(--muted)]">{w.rationale}</div>}
                              {w.metadata && Object.keys(w.metadata).length > 0 && !w.metadata.checks && (
                                <div className="mono break-all text-[11px] text-[var(--muted)]">{JSON.stringify(w.metadata)}</div>
                              )}
                            </td>
                            <td className={`tabular-nums ${w.value >= 1 ? "text-emerald-300" : w.value <= 0 ? "text-rose-300" : ""}`}>{num(w.value, 2)}</td>
                            <td>{w.source}</td>
                            <td className="tabular-nums">{w.step_index ?? "–"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </Panel>
                <Expander title="Raw record">
                  <JsonView value={rec} />
                </Expander>
                {p?.metadata && Object.keys(p.metadata).length > 0 && (
                  <Expander title="Payload metadata">
                    <JsonView value={p.metadata} />
                  </Expander>
                )}
                <div className="text-[12px] text-[var(--muted)]">
                  scores: {r.scores?.length ?? 0} · trace <IdLink href={`/traces/detail?id=${r.trace_id}`} id={r.trace_id} n={40} />
                </div>
              </div>
            </div>
          </>
        );
      }}
    </DataState>
  );
}

export default function Page() {
  return (
    <Suspense>
      <RolloutView />
    </Suspense>
  );
}
