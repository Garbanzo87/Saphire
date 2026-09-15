"use client";
import { useCallback, useEffect, useState } from "react";
import useSWR from "swr";
import { API_BASE, Api, ApiError, clearSession, setSession, type AuthConfig } from "@/lib/api";
import { ErrorBox, Field, Loading } from "@/components/ui";

function Card({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="panel p-4">
      <h2 className="text-[13px] font-semibold">{title}</h2>
      {hint && <p className="mt-0.5 text-[12px] text-[var(--muted)]">{hint}</p>}
      <div className="mt-3">{children}</div>
    </section>
  );
}

export default function LoginPage() {
  const { data: cfg, error: cfgErr } = useSWR<AuthConfig>("auth-config", () => Api.authConfig());
  const [email, setEmail] = useState("");
  const [devOrg, setDevOrg] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [keyOrg, setKeyOrg] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [fromCallback, setFromCallback] = useState(false);

  const finish = useCallback(async () => {
    try {
      await Api.me(); // validates the session; a 401 here lands back on this page
      window.location.href = "/";
    } catch (e) {
      setFromCallback(false);
      setErr(e instanceof ApiError && e.status === 401 ? `Sign-in rejected: ${e.message}` : e);
      setBusy(null);
    }
  }, []);

  // SSO callback lands here with #token=<jwt>
  useEffect(() => {
    const m = /(?:^#|&)token=([^&]+)/.exec(window.location.hash || "");
    if (!m) return;
    setFromCallback(true);
    clearSession();
    setSession({ token: decodeURIComponent(m[1]) });
    history.replaceState(null, "", window.location.pathname);
    finish();
  }, [finish]);

  async function devLogin(e: React.FormEvent) {
    e.preventDefault();
    setBusy("dev");
    setErr(null);
    try {
      clearSession();
      const { token } = await Api.devLogin({ email: email.trim(), org: devOrg.trim() || null });
      setSession({ token, org: devOrg.trim() || undefined });
      await finish();
    } catch (x) {
      setErr(x);
      setBusy(null);
    }
  }

  async function keyLogin(e: React.FormEvent) {
    e.preventDefault();
    setBusy("key");
    setErr(null);
    clearSession();
    setSession({ apiKey: apiKey.trim(), org: keyOrg.trim() || undefined });
    await finish();
  }

  const sso = cfg?.oidc ? cfg.login_url || `${API_BASE.replace(/\/$/, "")}/v1/auth/login` : null;

  return (
    <div className="mx-auto max-w-md py-10">
      <div className="mb-6 flex items-center gap-2">
        <span className="inline-block h-6 w-6 rotate-45 rounded-sm bg-gradient-to-br from-sky-400 to-violet-500" />
        <span className="text-lg font-semibold tracking-tight">Sign in to Saphire</span>
      </div>
      {fromCallback ? (
        <Loading label="Completing sign-in…" />
      ) : cfgErr ? (
        <ErrorBox error={cfgErr} />
      ) : !cfg ? (
        <Loading />
      ) : (
        <div className="space-y-4">
          {err ? <ErrorBox error={err} /> : null}
          {sso && (
            <Card title="Single sign-on" hint="Authenticate with your organization's identity provider.">
              <a href={sso} className="btn btn-primary inline-block">Sign in with SSO</a>
            </Card>
          )}
          {cfg.dev_login && (
            <Card title="Developer login" hint="Local development only — issues a session for any email without a password.">
              <form onSubmit={devLogin} className="space-y-3">
                <Field label="Email">
                  <input className="input" name="email" type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
                </Field>
                <Field label="Organization slug (optional)">
                  <input className="input" name="org" value={devOrg} onChange={(e) => setDevOrg(e.target.value)} placeholder="acme" />
                </Field>
                <button className="btn btn-primary" disabled={busy !== null || !email.trim()}>{busy === "dev" ? "Signing in…" : "Sign in"}</button>
              </form>
            </Card>
          )}
          <Card title="Use an API key" hint="Paste a root key or an organization key (sk_saph_…). Stored only in this browser.">
            <form onSubmit={keyLogin} className="space-y-3">
              <Field label="API key">
                <input className="input mono" name="apiKey" required autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk_saph_…" />
              </Field>
              <Field label="Organization slug (optional)">
                <input className="input" name="keyOrg" value={keyOrg} onChange={(e) => setKeyOrg(e.target.value)} placeholder="default" />
              </Field>
              <button className="btn btn-primary" disabled={busy !== null || !apiKey.trim()}>{busy === "key" ? "Checking…" : "Continue"}</button>
            </form>
          </Card>
          {!sso && !cfg.dev_login && <p className="text-[12px] text-[var(--muted)]">SSO and developer login are disabled on this server; use an API key.</p>}
        </div>
      )}
    </div>
  );
}
