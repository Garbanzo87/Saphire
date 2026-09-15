"use client";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { fetcher, q, type Agent, type Calibration, type ToolStats } from "@/lib/api";
import { ms, num, pct } from "@/lib/format";
import { MetricBarChart } from "@/components/charts";
import { Chip, DataState, KPI, PageHeader, Panel } from "@/components/ui";

function RateBar({ value, color }: { value: number | undefined; color: string }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 overflow-hidden rounded bg-[var(--border)]">
        <div className={`h-full ${color}`} style={{ width: `${v * 100}%` }} />
      </div>
      <span className="tabular-nums text-[12px]">{pct(value)}</span>
    </div>
  );
}

function CalibrationPanel() {
  const [judge, setJudge] = useState("judge_score");
  const [applied, setApplied] = useState("judge_score");
  const { data, error } = useSWR<Calibration>(q("/v1/signals/calibration", { judge: applied }), fetcher);
  const bins = useMemo(
    () =>
      Object.entries(data?.reliability || {})
        .sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))
        .map(([bin, v]) => ({ bin, human_positive_rate: v.human_positive_rate, n: v.n })),
    [data],
  );
  return (
    <Panel
      title="Judge calibration vs human labels"
      actions={
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setApplied(judge.trim() || "judge_score");
          }}
        >
          <input className="input w-56" value={judge} onChange={(e) => setJudge(e.target.value)} placeholder="judge_score or rm:<model>" aria-label="Judge" list="judge-options" />
          <datalist id="judge-options">
            <option value="judge_score" />
            <option value="rm:" />
          </datalist>
          <button className="btn">Load</button>
        </form>
      }
    >
      <DataState data={data} error={error}>
        {(c) =>
          c.n === 0 ? (
            <div className="text-[var(--muted)]">
              No paired labels for <span className="mono">{c.judge}</span>{c.note ? ` — ${c.note}` : ""}. Attach human or product scores to rollouts that also have this judge score to measure agreement.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
                <KPI label="Paired rollouts" value={c.paired_rollouts} hint={<span className="mono">{c.judge}</span>} />
                <KPI label="Agreement" value={pct(c.agreement)} hint={`@ threshold ${num(c.threshold, 2)}`} />
                <KPI label="Cohen's κ" value={num(c.kappa, 3)} hint={(c.kappa ?? 0) < 0.2 ? "slight" : (c.kappa ?? 0) < 0.4 ? "fair" : (c.kappa ?? 0) < 0.6 ? "moderate" : "substantial"} />
                <KPI label="Precision" value={pct(c.precision)} />
                <KPI label="Recall" value={pct(c.recall)} />
                <KPI label="F1" value={pct(c.f1)} />
                <KPI label="Best threshold" value={num(c.best_threshold, 2)} hint="maximises agreement" />
                <KPI label="Best agreement" value={pct(c.best_agreement)} hint={c.best_threshold !== undefined ? `@ ${num(c.best_threshold, 2)}` : undefined} />
              </div>
              <div className="grid gap-4 xl:grid-cols-2">
                <div>
                  <div className="label mb-2">Reliability diagram · human positive rate per judge-score bin</div>
                  {bins.length === 0 ? <div className="text-[var(--muted)]">No bins.</div> : <MetricBarChart data={bins} xKey="bin" series={[{ key: "human_positive_rate", label: "human positive rate" }]} height={220} />}
                </div>
                <div className="overflow-x-auto">
                  <table className="tbl">
                    <thead><tr><th>Judge score bin</th><th>n</th><th>Human positive rate</th><th>Calibration gap</th></tr></thead>
                    <tbody>
                      {bins.map((b) => {
                        const [lo, hi] = b.bin.split("-").map(Number);
                        const mid = Number.isFinite(lo) && Number.isFinite(hi) ? (lo + hi) / 2 : undefined;
                        const gap = mid !== undefined ? b.human_positive_rate - mid : undefined;
                        return (
                          <tr key={b.bin}>
                            <td className="mono text-[12px]">{b.bin}</td>
                            <td className="tabular-nums">{b.n}</td>
                            <td><RateBar value={b.human_positive_rate} color="bg-[var(--accent)]" /></td>
                            <td className={`tabular-nums ${gap === undefined ? "text-[var(--muted)]" : Math.abs(gap) > 0.2 ? "text-rose-300" : Math.abs(gap) > 0.1 ? "text-amber-300" : "text-emerald-300"}`}>{gap === undefined ? "–" : `${gap > 0 ? "+" : ""}${(gap * 100).toFixed(1)} pp`}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )
        }
      </DataState>
    </Panel>
  );
}

function ToolStatsPanel() {
  const { data: agents } = useSWR<Agent[]>("/v1/agents", fetcher);
  const names = Array.from(new Set((agents || []).map((a) => a.name)));
  const [agent, setAgent] = useState("");
  const { data, error } = useSWR<ToolStats>(q("/v1/signals/tool-stats", { agent_name: agent }), fetcher);
  const rows = useMemo(() => Object.entries(data || {}).sort((a, b) => b[1].calls - a[1].calls), [data]);
  const totalCalls = rows.reduce((s, [, v]) => s + v.calls, 0);
  return (
    <Panel
      title="Tool statistics"
      actions={
        <div className="flex items-center gap-2">
          {data && <span className="text-[11px] text-[var(--muted)]">{rows.length} tools · {totalCalls.toLocaleString()} calls</span>}
          <select className="input" value={agent} onChange={(e) => setAgent(e.target.value)} aria-label="Agent">
            <option value="">all agents</option>
            {names.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      }
    >
      <DataState data={data} error={error} isEmpty={(d) => Object.keys(d).length === 0} empty="No tool calls recorded.">
        {() => (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Tool</th><th>Calls</th><th>Share</th><th>Error rate</th><th>Success rate</th><th>Latency</th><th>Confused with</th></tr></thead>
              <tbody>
                {rows.map(([name, s]) => {
                  const confused = Object.entries(s.confused_with || {}).sort((a, b) => b[1] - a[1]);
                  return (
                    <tr key={name}>
                      <td className="mono text-[12px] font-medium">{name}</td>
                      <td className="tabular-nums">{s.calls.toLocaleString()}</td>
                      <td className="tabular-nums text-[var(--muted)]">{pct(totalCalls ? s.calls / totalCalls : undefined)}</td>
                      <td><RateBar value={s.error_rate} color={s.error_rate > 0.2 ? "bg-rose-400" : s.error_rate > 0.05 ? "bg-amber-400" : "bg-emerald-400"} /></td>
                      <td><RateBar value={s.success_rate} color="bg-[var(--accent)]" /></td>
                      <td className="tabular-nums">{ms(s.latency_ms)}</td>
                      <td>
                        {confused.length === 0 ? (
                          <span className="text-[var(--muted)]">–</span>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {confused.map(([other, n]) => <Chip key={other} status="4xx"><span className="mono">{other}</span> ×{n}</Chip>)}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </DataState>
    </Panel>
  );
}

export default function SignalsPage() {
  return (
    <>
      <PageHeader title="Signals" subtitle="How much to trust each reward signal: judge / reward-model calibration against human labels, and per-tool reliability across rollouts." />
      <div className="space-y-4">
        <CalibrationPanel />
        <ToolStatsPanel />
      </div>
    </>
  );
}
