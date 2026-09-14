"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { API_BASE, PROJECT } from "@/lib/api";

const nav = [
  ["/", "Overview"],
  ["/agents", "Agents"],
  ["/traces", "Traces"],
  ["/rollouts", "Rollouts"],
  ["/evals", "Evaluations"],
  ["/training", "Training"],
  ["/deployments", "Deployments"],
  ["/experiments", "Experiments"],
  ["/datasets", "Datasets"],
  ["/environments", "Environments"],
  ["/jobs", "Jobs"],
] as const;

export function Sidebar() {
  const path = usePathname() || "/";
  return (
    <aside className="flex h-full w-52 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--panel)]">
      <div className="flex items-center gap-2 px-4 py-4">
        <span className="inline-block h-5 w-5 rotate-45 rounded-sm bg-gradient-to-br from-sky-400 to-violet-500" />
        <span className="text-[15px] font-semibold tracking-tight">Saphire</span>
      </div>
      <nav className="flex flex-col gap-0.5 px-2">
        {nav.map(([href, label]) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`rounded-md px-2.5 py-1.5 text-[13px] ${active ? "bg-[var(--panel-2)] font-medium text-white" : "text-[var(--muted)] hover:bg-[var(--panel-2)] hover:text-white"}`}
            >
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto space-y-0.5 px-4 py-3 text-[11px] text-[var(--muted)]">
        <div>
          project <span className="mono text-[var(--fg)]">{PROJECT}</span>
        </div>
        <div className="truncate" title={API_BASE}>
          api <span className="mono">{API_BASE.replace(/^https?:\/\//, "")}</span>
        </div>
      </div>
    </aside>
  );
}
