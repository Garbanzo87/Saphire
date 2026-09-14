"use client";
import Link from "next/link";
import { useState, type ReactNode } from "react";
import { ApiError } from "@/lib/api";

const chipColors: Record<string, string> = {
  succeeded: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  deployed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  passed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  ok: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  running: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  pending: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  queued: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  candidate: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  failed: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  error: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  cancelled: "bg-zinc-500/15 text-zinc-300 border-zinc-500/30",
  retired: "bg-zinc-500/15 text-zinc-300 border-zinc-500/30",
  stopped: "bg-zinc-500/15 text-zinc-300 border-zinc-500/30",
  unset: "bg-zinc-500/15 text-zinc-300 border-zinc-500/30",
  agent: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  llm: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  tool: "bg-amber-500/15 text-amber-300 border-amber-500/30",
};

export function Chip({ status, children }: { status: string | boolean | null | undefined; children?: ReactNode }) {
  const s = status === true ? "passed" : status === false ? "failed" : String(status ?? "unset");
  const cls = chipColors[s] || "bg-zinc-500/15 text-zinc-300 border-zinc-500/30";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-4 ${cls}`}>
      {(s === "running" || s === "pending") && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {children ?? s}
    </span>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <div className="mt-1 text-[13px] text-[var(--muted)]">{subtitle}</div>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Panel({ title, children, className = "", actions }: { title?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-2.5">
          <h2 className="text-[13px] font-semibold">{title}</h2>
          {actions}
        </div>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-[var(--muted)]">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-[var(--muted)] border-t-transparent" />
      {label}
    </div>
  );
}

export function Empty({ children = "Nothing here yet." }: { children?: ReactNode }) {
  return <div className="py-10 text-center text-[var(--muted)]">{children}</div>;
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof ApiError ? `${error.status}: ${error.message}` : error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-[13px] text-rose-200">
      Request failed — {msg}. Check that the API is reachable and CORS allows this origin.
    </div>
  );
}

/** Wraps loading / error / empty states for SWR results. */
export function DataState<T>({
  data,
  error,
  isEmpty,
  empty,
  children,
}: {
  data: T | undefined;
  error: unknown;
  isEmpty?: (d: T) => boolean;
  empty?: ReactNode;
  children: (d: T) => ReactNode;
}) {
  if (error) return <ErrorBox error={error} />;
  if (data === undefined) return <Loading />;
  if (isEmpty && isEmpty(data)) return <Empty>{empty}</Empty>;
  return <>{children(data)}</>;
}

export function KPI({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="panel px-4 py-3">
      <div className="label">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="mt-0.5 text-[12px] text-[var(--muted)]">{hint}</div>}
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--panel-2)] px-3 py-2">
      <div className="label">{label}</div>
      <div className="mt-0.5 text-[15px] font-semibold tabular-nums">{value}</div>
    </div>
  );
}

export function IdLink({ href, id, n = 16 }: { href: string; id: string | null | undefined; n?: number }) {
  if (!id) return <span className="text-[var(--muted)]">–</span>;
  return (
    <Link href={href} className="link mono text-[12px]" title={id}>
      {id.length > n ? id.slice(0, n) + "…" : id}
    </Link>
  );
}

export function Expander({ title, children, defaultOpen = false }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-md border border-[var(--border)]">
      <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] hover:bg-[var(--panel-2)]">
        <span className="text-[var(--muted)]">{open ? "▾" : "▸"}</span>
        {title}
      </button>
      {open && <div className="border-t border-[var(--border)] p-3">{children}</div>}
    </div>
  );
}

export function Progress({ value }: { value: number | null | undefined }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-28 overflow-hidden rounded bg-[var(--border)]">
        <div className="h-full bg-[var(--accent)]" style={{ width: `${v * 100}%` }} />
      </div>
      <span className="tabular-nums text-[12px] text-[var(--muted)]">{Math.round(v * 100)}%</span>
    </div>
  );
}

export function Logs({ logs }: { logs: string | null | undefined }) {
  if (!logs) return <Empty>No logs.</Empty>;
  return <pre className="mono max-h-96 overflow-auto rounded-md bg-[var(--bg)] p-3 text-[12px] leading-5 text-[var(--fg)]">{logs}</pre>;
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}
