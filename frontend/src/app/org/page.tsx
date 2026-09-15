"use client";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { Api, fetcher, q, type ApiKey, type Me, type Member, type Org, type Usage } from "@/lib/api";
import { ago, ts } from "@/lib/format";
import { MetricLineChart } from "@/components/charts";
import { Chip, DataState, ErrorBox, Field, KPI, PageHeader, Panel } from "@/components/ui";

const USAGE_METRICS: { key: string; label: string }[] = [
  { key: "rollouts", label: "Rollouts" },
  { key: "spans", label: "Spans" },
  { key: "tokens", label: "Tokens" },
  { key: "jobs", label: "Jobs" },
  { key: "api_requests", label: "API requests" },
];
const ROLES = ["viewer", "member", "admin", "owner"];
const PLANS = ["free", "team", "enterprise"];
const fmtInt = (v: unknown) => (typeof v === "number" ? Math.round(v).toLocaleString() : "0");

function QuotaBar({ label, used, limit }: { label: string; used: number; limit: number }) {
  const ratio = limit > 0 ? used / limit : 0;
  const color = ratio >= 1 ? "bg-rose-400" : ratio >= 0.8 ? "bg-amber-400" : "bg-[var(--accent)]";
  return (
    <div>
      <div className="flex justify-between text-[12px]">
        <span>{label}</span>
        <span className="tabular-nums text-[var(--muted)]">
          {fmtInt(used)} / {fmtInt(limit)} · {Math.round(ratio * 100)}%
        </span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded bg-[var(--border)]">
        <div className={`h-full ${color}`} style={{ width: `${Math.min(100, ratio * 100)}%` }} />
      </div>
    </div>
  );
}

function UsagePanel({ usage }: { usage: Usage }) {
  const [metric, setMetric] = useState("rollouts");
  const [days, setDays] = useState(30);
  const { data, error } = useSWR<Usage>(q("/v1/orgs/current/usage", { days }), fetcher, { fallbackData: days === 30 ? usage : undefined });
  const rows = useMemo(() => {
    const d = data?.daily || {};
    return Object.keys(d)
      .sort()
      .map((day) => ({ day: day.slice(5), ...Object.fromEntries(USAGE_METRICS.map((m) => [m.key, d[day]?.[m.key] ?? 0])) }));
  }, [data]);
  const series = [{ key: metric, label: USAGE_METRICS.find((m) => m.key === metric)?.label }];
  const today = data?.daily?.[Object.keys(data?.daily || {}).sort().pop() || ""] || {};
  const totals = data?.totals || {};
  const quotas = data?.quotas || {};
  return (
    <Panel
      title={`Usage · last ${data?.days ?? days} days`}
      actions={
        <div className="flex items-center gap-2">
          <select className="input" value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Window">
            {[7, 30, 90].map((d) => <option key={d} value={d}>{d} days</option>)}
          </select>
          <select className="input" value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="Metric">
            {USAGE_METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>
      }
    >
      <DataState data={data} error={error}>
        {() => (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {USAGE_METRICS.map((m) => <KPI key={m.key} label={m.label} value={fmtInt(totals[m.key])} hint={`today ${fmtInt(today[m.key])}`} />)}
            </div>
            {rows.length === 0 ? (
              <div className="text-[var(--muted)]">No usage recorded in this window.</div>
            ) : (
              <MetricLineChart data={rows} xKey="day" series={series} percent={false} height={240} />
            )}
            {(quotas.rollouts_per_month !== undefined || quotas.jobs_per_day !== undefined) && (
              <div className="grid gap-3 md:grid-cols-2">
                {quotas.rollouts_per_month !== undefined && <QuotaBar label="Rollouts this month" used={totals.rollouts ?? 0} limit={quotas.rollouts_per_month} />}
                {quotas.jobs_per_day !== undefined && <QuotaBar label="Jobs today" used={today.jobs ?? 0} limit={quotas.jobs_per_day} />}
              </div>
            )}
          </div>
        )}
      </DataState>
    </Panel>
  );
}

function MembersPanel({ canManage }: { canManage: boolean }) {
  const { data, error, mutate } = useSWR<Member[]>("/v1/orgs/current/members", fetcher);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("member");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    setBusy("add");
    setErr(null);
    try {
      await Api.addMember({ email: email.trim(), role });
      setEmail("");
      await mutate();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(null);
    }
  }
  async function remove(m: Member) {
    if (!confirm(`Remove ${m.email} from the organization?`)) return;
    setBusy(m.membership_id);
    setErr(null);
    try {
      await Api.removeMember(m.membership_id);
      await mutate();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(null);
    }
  }
  return (
    <Panel title="Members">
      <div className="space-y-3">
        {canManage && (
          <form onSubmit={add} className="flex flex-wrap items-end gap-2">
            <Field label="Email">
              <input className="input w-64" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="person@company.com" />
            </Field>
            <Field label="Role">
              <select className="input" value={role} onChange={(e) => setRole(e.target.value)}>
                {ROLES.map((r) => <option key={r}>{r}</option>)}
              </select>
            </Field>
            <button className="btn btn-primary" disabled={busy !== null || !email.trim()}>{busy === "add" ? "Adding…" : "Add member"}</button>
          </form>
        )}
        {err ? <ErrorBox error={err} /> : null}
        <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No members yet.">
          {(rows) => (
            <div className="overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr><th>Email</th><th>Name</th><th>Role</th><th>SSO</th><th>Last login</th><th>Joined</th>{canManage && <th></th>}</tr>
                </thead>
                <tbody>
                  {rows.map((m) => (
                    <tr key={m.membership_id}>
                      <td className="font-medium">{m.email}</td>
                      <td>{m.name || <span className="text-[var(--muted)]">–</span>}</td>
                      <td><Chip status="candidate">{m.role}</Chip></td>
                      <td>{m.sso_provider || <span className="text-[var(--muted)]">–</span>}</td>
                      <td className="whitespace-nowrap text-[var(--muted)]" title={m.last_login_at ? ts(m.last_login_at) : ""}>{m.last_login_at ? ago(m.last_login_at) : "never"}</td>
                      <td className="whitespace-nowrap text-[var(--muted)]">{ts(m.created_at)}</td>
                      {canManage && (
                        <td className="text-right">
                          <button className="btn" disabled={busy !== null} onClick={() => remove(m)}>{busy === m.membership_id ? "Removing…" : "Remove"}</button>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </DataState>
      </div>
    </Panel>
  );
}

function KeysPanel({ canManage }: { canManage: boolean }) {
  const { data, error, mutate } = useSWR<ApiKey[]>("/v1/orgs/current/keys", fetcher);
  const [name, setName] = useState("");
  const [role, setRole] = useState("member");
  const [expires, setExpires] = useState("");
  const [created, setCreated] = useState<ApiKey | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setBusy("create");
    setErr(null);
    setCreated(null);
    try {
      const k = await Api.createKey({ name: name.trim(), role, expires_in_days: expires ? Number(expires) : null });
      setCreated(k);
      setName("");
      await mutate();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(null);
    }
  }
  async function revoke(k: ApiKey) {
    if (!confirm(`Revoke key "${k.name}" (${k.prefix}…)? Clients using it will start receiving 401.`)) return;
    setBusy(k.id);
    setErr(null);
    try {
      await Api.revokeKey(k.id);
      await mutate();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(null);
    }
  }
  async function copy() {
    try {
      await navigator.clipboard.writeText(created?.key || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }
  return (
    <Panel title="API keys">
      <div className="space-y-3">
        {canManage && (
          <form onSubmit={create} className="flex flex-wrap items-end gap-2">
            <Field label="Name">
              <input className="input w-56" required value={name} onChange={(e) => setName(e.target.value)} placeholder="ci key" />
            </Field>
            <Field label="Role">
              <select className="input" value={role} onChange={(e) => setRole(e.target.value)}>
                {ROLES.map((r) => <option key={r}>{r}</option>)}
              </select>
            </Field>
            <Field label="Expires in days">
              <input className="input w-28" type="number" min={1} value={expires} onChange={(e) => setExpires(e.target.value)} placeholder="never" />
            </Field>
            <button className="btn btn-primary" disabled={busy !== null || !name.trim()}>{busy === "create" ? "Creating…" : "Create key"}</button>
          </form>
        )}
        {created?.key && (
          <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-[13px]">
            <div className="mb-1 font-medium text-emerald-200">Key &ldquo;{created.name}&rdquo; created — copy it now, it will not be shown again.</div>
            <div className="flex items-center gap-2">
              <code className="mono flex-1 select-all overflow-x-auto rounded bg-[var(--bg)] px-2 py-1 text-[12px]">{created.key}</code>
              <button type="button" className="btn" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
              <button type="button" className="btn" onClick={() => setCreated(null)}>Dismiss</button>
            </div>
          </div>
        )}
        {err ? <ErrorBox error={err} /> : null}
        <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No API keys.">
          {(rows) => (
            <div className="overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr><th>Name</th><th>Prefix</th><th>Role</th><th>Created</th><th>Last used</th><th>Expires</th><th>Status</th>{canManage && <th></th>}</tr>
                </thead>
                <tbody>
                  {rows.map((k) => {
                    const expired = k.expires_at !== null && k.expires_at < Date.now() / 1000;
                    const state = k.revoked_at ? "revoked" : expired ? "expired" : "active";
                    return (
                      <tr key={k.id} className={state !== "active" ? "opacity-60" : ""}>
                        <td className="font-medium">{k.name}<div className="mono text-[11px] text-[var(--muted)]">{k.id}</div></td>
                        <td className="mono text-[12px]">{k.prefix}…</td>
                        <td><Chip status="candidate">{k.role}</Chip></td>
                        <td className="whitespace-nowrap text-[var(--muted)]">{ts(k.created_at)}<div className="text-[11px]">by {k.created_by}</div></td>
                        <td className="whitespace-nowrap text-[var(--muted)]">{k.last_used_at ? ago(k.last_used_at) : "never"}</td>
                        <td className="whitespace-nowrap text-[var(--muted)]">{k.expires_at ? ts(k.expires_at) : "never"}</td>
                        <td><Chip status={state === "active" ? "ok" : state === "revoked" ? "failed" : "retired"}>{state}</Chip></td>
                        {canManage && (
                          <td className="text-right">
                            {!k.revoked_at && <button className="btn" disabled={busy !== null} onClick={() => revoke(k)}>{busy === k.id ? "Revoking…" : "Revoke"}</button>}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </DataState>
      </div>
    </Panel>
  );
}

function SettingsPanel({ org, onSaved }: { org: Org; onSaved: () => void }) {
  const [plan, setPlan] = useState(org.plan);
  const [domains, setDomains] = useState((org.settings?.allowed_email_domains || []).join(", "));
  const [retention, setRetention] = useState(String(org.settings?.retention_days ?? 0));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setSaved(false);
    try {
      await Api.patchOrg({
        plan,
        settings: {
          ...(org.settings || {}),
          allowed_email_domains: domains.split(",").map((d) => d.trim().toLowerCase()).filter(Boolean),
          retention_days: Math.max(0, Number(retention) || 0),
        },
      });
      setSaved(true);
      onSaved();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Panel title="Settings" actions={<span className="text-[11px] text-[var(--muted)]">owner only</span>}>
      <form onSubmit={save} className="space-y-3">
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Plan">
            <select className="input" value={plan} onChange={(e) => setPlan(e.target.value)}>
              {(PLANS.includes(plan) ? PLANS : [...PLANS, plan]).map((p) => <option key={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Allowed email domains (comma-separated)">
            <input className="input" value={domains} onChange={(e) => setDomains(e.target.value)} placeholder="acme.com, acme.io" />
          </Field>
          <Field label="Retention days (0 = keep forever)">
            <input className="input" type="number" min={0} value={retention} onChange={(e) => setRetention(e.target.value)} />
          </Field>
        </div>
        <div className="flex items-center gap-3">
          <button className="btn btn-primary" disabled={busy}>{busy ? "Saving…" : "Save settings"}</button>
          {saved && <span className="text-[12px] text-emerald-300">Saved.</span>}
        </div>
        {err ? <ErrorBox error={err} /> : null}
      </form>
    </Panel>
  );
}

export default function OrgPage() {
  const { data: org, error, mutate } = useSWR<Org>("/v1/orgs/current", fetcher);
  const { data: me } = useSWR<Me>("auth-me", () => Api.me(), { shouldRetryOnError: false });
  const { data: usage, error: usageErr } = useSWR<Usage>(q("/v1/orgs/current/usage", { days: 30 }), fetcher);
  const perms = new Set(me?.permissions || []);
  const canMembers = perms.has("members");
  const canKeys = perms.has("keys");
  const canWrite = perms.has("org.write");
  return (
    <>
      <PageHeader
        title="Organization"
        subtitle={org ? <span className="flex flex-wrap items-center gap-2"><span className="font-medium text-[var(--fg)]">{org.name}</span> · slug <span className="mono">{org.slug}</span> · <Chip status="candidate">{org.plan}</Chip> · {org.n_members} members · {org.n_api_keys} keys · {org.n_projects} projects · created {ts(org.created_at)}</span> : undefined}
        actions={org?.you ? <span className="text-[12px] text-[var(--muted)]">you: {org.you.label} · {org.you.role}</span> : null}
      />
      <DataState data={org} error={error}>
        {(o) => {
          const quotas = Object.entries(o.effective_quotas || {});
          return (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                {quotas.length === 0 ? (
                  <KPI label="Quotas" value="unlimited" hint={`${o.plan} plan`} />
                ) : (
                  quotas.map(([k, v]) => <KPI key={k} label={k.replace(/_/g, " ")} value={fmtInt(v)} hint={(o.quotas || {})[k] !== undefined ? "org override" : `${o.plan} plan`} />)
                )}
              </div>
              {usageErr ? <ErrorBox error={usageErr} /> : usage ? <UsagePanel usage={usage} /> : null}
              <MembersPanel canManage={canMembers} />
              <KeysPanel canManage={canKeys} />
              {canWrite && <SettingsPanel key={`${o.plan}|${JSON.stringify(o.settings)}`} org={o} onSaved={() => mutate()} />}
            </div>
          );
        }}
      </DataState>
    </>
  );
}
