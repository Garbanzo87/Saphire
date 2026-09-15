"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import { API_BASE, Api, PROJECT, clearSession, getSession, setSession, type Me, type Org } from "@/lib/api";
import { Chip } from "@/components/ui";

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
  ["/intelligence", "Intelligence"],
  ["/attribution", "Attribution"],
  ["/signals", "Signals"],
  ["/webhooks", "Webhooks"],
  ["/org", "Organization"],
  ["/audit", "Audit log"],
] as const;

function Principal() {
  const { data: me, error } = useSWR<Me>("auth-me", () => Api.me(), { shouldRetryOnError: false });
  const canSwitch = !!me && (me.role === "superadmin" || (me.orgs?.length ?? 0) > 1);
  const { data: orgs } = useSWR<Org[]>(canSwitch ? "orgs" : null, () => Api.orgs(), { shouldRetryOnError: false });
  const current = getSession().org || me?.org || "";

  function signOut() {
    clearSession();
    window.location.href = "/login/";
  }
  function switchOrg(slug: string) {
    setSession({ org: slug || undefined });
    window.location.reload();
  }

  if (error)
    return (
      <div className="space-y-1">
        <div className="text-rose-300">not signed in</div>
        <Link href="/login" className="link">Sign in →</Link>
      </div>
    );
  if (!me) return <div className="text-[var(--muted)]">…</div>;
  const options: { slug: string; name: string }[] = orgs ? orgs.map((o) => ({ slug: o.slug, name: o.name })) : (me.orgs || []).map((o) => ({ slug: o.slug, name: o.name }));
  if (current && !options.some((o) => o.slug === current)) options.unshift({ slug: current, name: current });
  return (
    <div className="space-y-1.5" data-testid="principal">
      <div className="truncate text-[12px] font-medium text-[var(--fg)]" title={me.label}>{me.label}</div>
      <div className="flex flex-wrap items-center gap-1">
        <Chip status={me.actor_type}>{me.actor_type}</Chip>
        <Chip status="candidate">{me.role}</Chip>
      </div>
      <div>
        org{" "}
        {canSwitch ? (
          <select className="input ml-1 py-0.5 text-[11px]" value={current} onChange={(e) => switchOrg(e.target.value)} aria-label="Organization">
            {!current && <option value="">(default)</option>}
            {options.map((o) => (
              <option key={o.slug} value={o.slug}>{o.name} · {o.slug}</option>
            ))}
          </select>
        ) : (
          <span className="mono text-[var(--fg)]">{me.org || "–"}</span>
        )}
      </div>
      <button type="button" onClick={signOut} className="link">Sign out</button>
    </div>
  );
}

export function Sidebar() {
  const path = usePathname() || "/";
  if (path.startsWith("/login")) return null;
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
      <div className="mt-auto space-y-2 border-t border-[var(--border)] px-4 py-3 text-[11px] text-[var(--muted)]">
        <Principal />
        <div className="space-y-0.5 border-t border-[var(--border)] pt-2">
          <div>
            project <span className="mono text-[var(--fg)]">{PROJECT}</span>
          </div>
          <div className="truncate" title={API_BASE}>
            api <span className="mono">{API_BASE.replace(/^https?:\/\//, "")}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
