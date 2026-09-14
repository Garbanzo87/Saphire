export const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function pct(v: unknown, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : "–";
}
export function num(v: unknown, digits = 2): string {
  return isNum(v) ? v.toFixed(digits) : "–";
}
export function ms(v: unknown): string {
  if (!isNum(v)) return "–";
  if (v >= 60_000) return `${(v / 60_000).toFixed(1)} min`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(v < 10 ? 2 : 0)} ms`;
}
export function delta(v: unknown, digits = 1): string {
  if (!isNum(v)) return "–";
  const s = (v * 100).toFixed(digits);
  return v > 0 ? `+${s} pp` : `${s} pp`;
}
export function pval(v: unknown): string {
  if (!isNum(v)) return "–";
  return v < 0.001 ? "<0.001" : v.toFixed(3);
}
export function ts(v: unknown): string {
  if (!isNum(v)) return "–";
  const d = new Date(v * 1000);
  return d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
export function ago(v: unknown): string {
  if (!isNum(v)) return "–";
  const s = Math.max(0, Date.now() / 1000 - v);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
export function short(id: string | null | undefined, n = 12) {
  if (!id) return "–";
  return id.length > n ? id.slice(0, n) + "…" : id;
}
/** Try to parse a string as JSON; return the parsed value or the original string. */
export function tryJson(v: unknown): unknown {
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!(t.startsWith("{") || t.startsWith("["))) return v;
  try {
    return JSON.parse(t);
  } catch {
    return v;
  }
}
export const isActive = (status?: string | null) => status === "running" || status === "pending" || status === "queued";
