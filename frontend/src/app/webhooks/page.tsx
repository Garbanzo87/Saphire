"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { Api, ApiError, fetcher, q, type Webhook, type WebhookDelivery } from "@/lib/api";
import { ago, ts } from "@/lib/format";
import { JsonView } from "@/components/JsonView";
import { Chip, DataState, ErrorBox, Expander, Field, PageHeader, Panel } from "@/components/ui";

const statusClass = (code: number) => (code >= 200 && code < 300 ? "2xx" : code >= 300 && code < 400 ? "3xx" : code >= 400 && code < 500 ? "4xx" : code >= 500 ? "5xx" : "failed");

function StatusChip({ code, error }: { code: number; error?: string }) {
  return <Chip status={statusClass(code)}>{code > 0 ? code : error ? "no response" : "–"}</Chip>;
}

function CreateForm({ events, onCreated }: { events: string[]; onCreated: (w: Webhook) => void }) {
  const [form, setForm] = useState({ url: "", description: "", secret: "", events: new Set<string>(events) });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  function toggle(ev: string) {
    const next = new Set(form.events);
    if (next.has(ev)) next.delete(ev);
    else next.add(ev);
    setForm({ ...form, events: next });
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const w = await Api.createWebhook({ url: form.url.trim(), events: Array.from(form.events), description: form.description.trim(), ...(form.secret.trim() ? { secret: form.secret.trim() } : {}) });
      setForm({ ...form, url: "", description: "", secret: "" });
      onCreated(w);
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="grid gap-3 md:grid-cols-3">
        <Field label="URL">
          <input className="input" type="url" required value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} placeholder="https://hooks.example.com/saphire" />
        </Field>
        <Field label="Description">
          <input className="input" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} placeholder="alerts to #ops" />
        </Field>
        <Field label="Signing secret (optional, generated if empty)">
          <input className="input" value={form.secret} onChange={(e) => setForm({ ...form, secret: e.target.value })} placeholder="auto" />
        </Field>
      </div>
      <div>
        <div className="label mb-1.5">Events</div>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[13px]">
          {events.map((ev) => (
            <label key={ev} className="flex items-center gap-2">
              <input type="checkbox" checked={form.events.has(ev)} onChange={() => toggle(ev)} /> <span className="mono text-[12px]">{ev}</span>
            </label>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <button className="btn btn-primary" disabled={busy || !form.url.trim() || form.events.size === 0}>{busy ? "Creating…" : "Create webhook"}</button>
        <span className="text-[12px] text-[var(--muted)]">Deliveries are signed with HMAC-SHA256 over the raw body.</span>
      </div>
      {err ? <ErrorBox error={err} /> : null}
    </form>
  );
}

function DeliveriesTable({ rows }: { rows: WebhookDelivery[] }) {
  if (rows.length === 0) return <div className="text-[var(--muted)]">No deliveries yet.</div>;
  return (
    <div className="max-h-[24rem] overflow-auto">
      <table className="tbl">
        <thead><tr><th>Delivery</th><th>Event</th><th>Status</th><th>Attempts</th><th>Result</th><th>Payload</th><th>Created</th></tr></thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.id}>
              <td className="mono text-[12px]">{d.id}</td>
              <td className="mono text-[12px]">{d.event}</td>
              <td><StatusChip code={d.status_code} error={d.error} /></td>
              <td className="tabular-nums">{d.attempts}</td>
              <td className={`max-w-[24rem] truncate text-[12px] ${d.status_code >= 200 && d.status_code < 300 ? "text-emerald-300" : "text-rose-300"}`} title={d.error}>{d.status_code >= 200 && d.status_code < 300 ? "ok" : d.error || "failed"}</td>
              <td>
                <Expander title={<span className="text-[12px] text-[var(--muted)]">{Object.keys(d.payload || {}).length} keys</span>}>
                  <JsonView value={d.payload} />
                </Expander>
              </td>
              <td className="whitespace-nowrap text-[var(--muted)]" title={ts(d.created_at)}>{ago(d.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WebhookRow({ w, onChanged }: { w: Webhook; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [testResult, setTestResult] = useState<WebhookDelivery[] | null>(null);
  const { data: deliveries, error: delErr, mutate } = useSWR<WebhookDelivery[]>(open ? q(`/v1/orgs/current/webhooks/${w.id}/deliveries`, { limit: 50 }) : null, fetcher);

  async function test() {
    setBusy("test");
    setErr(null);
    try {
      setTestResult(await Api.testWebhook(w.id));
      setOpen(true);
      await mutate();
      onChanged();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(null);
    }
  }
  async function remove() {
    if (!confirm(`Delete webhook ${w.id} (${w.url})?`)) return;
    setBusy("delete");
    setErr(null);
    try {
      await Api.deleteWebhook(w.id);
      onChanged();
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(null);
    }
  }
  const last = w.last_delivery;
  return (
    <div className="panel">
      <div className="flex flex-wrap items-start justify-between gap-3 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="mono text-[12px] text-[var(--muted)]">{w.id}</span>
            <Chip status={w.active === false ? "retired" : "ok"}>{w.active === false ? "inactive" : "active"}</Chip>
            <span className="mono truncate text-[13px] font-medium" title={w.url}>{w.url}</span>
          </div>
          {w.description && <div className="mt-0.5 text-[12px] text-[var(--muted)]">{w.description}</div>}
          <div className="mt-1.5 flex flex-wrap gap-1">
            {w.events.map((e) => <Chip key={e} status="candidate">{e}</Chip>)}
          </div>
          <div className="mt-1.5 text-[12px] text-[var(--muted)]">
            created {ts(w.created_at)} · last delivery{" "}
            {last ? (
              <>
                <StatusChip code={last.status_code} error={last.error} /> <span className="mono">{last.event}</span> {ago(last.created_at)}
              </>
            ) : (
              "never"
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn" disabled={busy !== null} onClick={test}>{busy === "test" ? "Sending…" : "Send test"}</button>
          <button className="btn" onClick={() => setOpen(!open)}>{open ? "Hide deliveries" : "Deliveries"}</button>
          <button className="btn" disabled={busy !== null} onClick={remove}>{busy === "delete" ? "Deleting…" : "Delete"}</button>
        </div>
      </div>
      {(err || testResult || open) && (
        <div className="space-y-3 border-t border-[var(--border)] px-4 py-3">
          {err ? <ErrorBox error={err} /> : null}
          {testResult && (
            <div className={`rounded-md border px-3 py-2 text-[13px] ${testResult.some((d) => d.status_code >= 200 && d.status_code < 300) ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200" : "border-amber-500/40 bg-amber-500/10 text-amber-100"}`}>
              {testResult.length === 0 ? (
                <>No delivery made — this webhook is not subscribed to the <span className="mono">test</span> event (only hooks subscribed to it, or to all events, receive tests).</>
              ) : (
                <>Test event sent · {testResult.length} delivery attempt{testResult.length === 1 ? "" : "s"}:{" "}</>
              )}
              {testResult.map((d) => <span key={d.id} className="mr-2"><StatusChip code={d.status_code} error={d.error} /> <span className="mono text-[11px]">{d.error ? d.error.slice(0, 80) : "ok"}</span></span>)}
            </div>
          )}
          {open && (
            <DataState data={deliveries} error={delErr}>
              {(rows) => <DeliveriesTable rows={rows} />}
            </DataState>
          )}
        </div>
      )}
    </div>
  );
}

export default function WebhooksPage() {
  const { data, error, mutate } = useSWR<Webhook[]>("/v1/orgs/current/webhooks", fetcher, { shouldRetryOnError: false });
  const { data: events } = useSWR<string[]>("/v1/orgs/current/webhooks/events", fetcher);
  const [showForm, setShowForm] = useState(false);
  const [created, setCreated] = useState<Webhook | null>(null);
  const forbidden = error instanceof ApiError && (error.status === 401 || error.status === 403);
  return (
    <>
      <PageHeader
        title="Webhooks"
        subtitle={data ? `${data.length} webhooks · org-level event subscriptions (job, gate, quota, intelligence, promotion events)` : "Org-level event subscriptions."}
        actions={<button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>{showForm ? "Close" : "New webhook"}</button>}
      />
      {showForm && (
        <Panel title="New webhook" className="mb-4">
          <CreateForm
            events={events || []}
            onCreated={(w) => {
              setCreated(w);
              setShowForm(false);
              mutate();
            }}
          />
        </Panel>
      )}
      {created?.secret && (
        <div className="mb-4 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-[13px]">
          <div className="mb-1 font-medium text-emerald-200">Webhook {created.id} created — copy the signing secret now, it will not be shown again.</div>
          <div className="flex items-center gap-2">
            <code className="mono flex-1 select-all overflow-x-auto rounded bg-[var(--bg)] px-2 py-1 text-[12px]">{created.secret}</code>
            <button type="button" className="btn" onClick={() => setCreated(null)}>Dismiss</button>
          </div>
        </div>
      )}
      {forbidden ? (
        <div className="space-y-1 rounded-md border border-[var(--border)] bg-[var(--panel)] px-4 py-6 text-center text-[var(--muted)]">
          <div>Managing webhooks needs the <span className="mono">keys</span> permission on this organization (admin or owner).</div>
          <Link href="/login" className="link">Sign in →</Link>
        </div>
      ) : (
        <DataState data={data} error={error} isEmpty={(d) => d.length === 0} empty="No webhooks yet — create one above.">
          {(rows) => (
            <div className="space-y-3">
              {rows.map((w) => <WebhookRow key={w.id} w={w} onChanged={() => mutate()} />)}
            </div>
          )}
        </DataState>
      )}
    </>
  );
}
