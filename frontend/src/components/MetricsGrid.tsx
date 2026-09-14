"use client";
import type { Metrics } from "@/lib/api";
import { ms, num, pct } from "@/lib/format";
import { Stat } from "@/components/ui";

const DEFS: { key: keyof Metrics; label: string; fmt: (v: unknown) => string }[] = [
  { key: "task_success", label: "Task success", fmt: (v) => pct(v) },
  { key: "tool_selection_f1", label: "Tool selection F1", fmt: (v) => pct(v) },
  { key: "context_preservation", label: "Context preservation", fmt: (v) => pct(v) },
  { key: "step_efficiency", label: "Step efficiency", fmt: (v) => pct(v) },
  { key: "tool_error_free", label: "Tool error-free", fmt: (v) => pct(v) },
  { key: "judge_score", label: "Judge score", fmt: (v) => pct(v) },
  { key: "pass_at_1", label: "pass@1", fmt: (v) => pct(v) },
  { key: "consistency", label: "Consistency", fmt: (v) => pct(v) },
  { key: "error_rate", label: "Error rate", fmt: (v) => pct(v) },
  { key: "latency_ms_p50", label: "Latency p50", fmt: (v) => ms(v) },
  { key: "latency_ms_p95", label: "Latency p95", fmt: (v) => ms(v) },
  { key: "throughput_tasks_per_min", label: "Throughput /min", fmt: (v) => num(v, 0) },
  { key: "steps_mean", label: "Mean steps", fmt: (v) => num(v, 2) },
  { key: "tokens_per_task", label: "Tokens / task", fmt: (v) => num(v, 0) },
  { key: "n", label: "n", fmt: (v) => num(v, 0) },
];

export function MetricsGrid({ m }: { m: Metrics | null | undefined }) {
  if (!m) return <div className="text-[var(--muted)]">No metrics yet.</div>;
  const extra = Object.keys(m).filter((k) => /^pass_(at|hat)_\d+$/.test(k) && k !== "pass_at_1");
  const items = [...DEFS.filter((d) => m[d.key] !== undefined && m[d.key] !== null), ...extra.map((k) => ({ key: k, label: k.replace(/^pass_hat_(\d+)$/, "pass^$1").replace(/^pass_at_(\d+)$/, "pass@$1"), fmt: (v: unknown) => pct(v) }))];
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-5">
      {items.map((d) => (
        <Stat
          key={String(d.key)}
          label={d.label}
          value={
            <>
              {d.fmt(m[d.key as string])}
              {d.key === "task_success" && m.task_success_ci_low !== undefined && (
                <span className="ml-1 text-[11px] font-normal text-[var(--muted)]">
                  [{pct(m.task_success_ci_low, 0)}–{pct(m.task_success_ci_high, 0)}]
                </span>
              )}
            </>
          }
        />
      ))}
    </div>
  );
}
