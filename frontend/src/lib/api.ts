// Small typed client for the Saphire API. All calls are made from the browser.
export const API_BASE = process.env.NEXT_PUBLIC_SAPHIRE_API || "http://localhost:8000";
export const API_KEY = process.env.NEXT_PUBLIC_SAPHIRE_API_KEY || "dev-key";
export const PROJECT = process.env.NEXT_PUBLIC_SAPHIRE_PROJECT || "default";

export type Metrics = Record<string, number | undefined> & {
  n?: number;
  task_success?: number;
  tool_selection_f1?: number;
  step_efficiency?: number;
  context_preservation?: number;
  tool_error_free?: number;
  latency_ms_p50?: number;
  latency_ms_p95?: number;
  steps_mean?: number;
  tokens_per_task?: number;
  error_rate?: number;
  pass_at_1?: number;
  throughput_tasks_per_min?: number;
  task_success_ci_low?: number;
  task_success_ci_high?: number;
};

export interface AgentConfig {
  name: string;
  version: string;
  model: string;
  system_prompt: string;
  temperature?: number;
  max_steps?: number;
  tool_router?: string | null;
  router_top_k?: number | null;
  exemplar_store?: string | null;
  exemplar_k?: number | null;
  roles?: Record<string, RoleConfig>;
  orchestrator_prompt?: string | null;
  orchestrator_tools?: string[];
  metadata?: Record<string, unknown>;
  [k: string]: unknown;
}
export interface RoleConfig {
  description?: string;
  system_prompt?: string | null;
  model?: string | null;
  tool_names?: string[];
  tool_tags?: string[];
  tool_router?: string | null;
  router_top_k?: number | null;
  exemplar_store?: string | null;
  exemplar_k?: number | null;
  max_steps?: number;
  trainable?: boolean;
  temperature?: number;
  [k: string]: unknown;
}
export type RoleMetrics = { step_reward?: number; steps_per_rollout?: number; tool_calls_per_rollout?: number; n?: number; [k: string]: number | undefined };
export interface Agent {
  id: string;
  project_id: string;
  name: string;
  version: string;
  config: AgentConfig;
  parent_id: string | null;
  status: string;
  origin: string;
  created_at: number;
}

export interface Job {
  id: string;
  type: string;
  status: string;
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  progress: number;
  logs: string;
  error: string;
  worker?: string;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface EvalRun {
  id: string;
  agent_id: string;
  dataset_id: string;
  suite: string;
  k: number;
  judge_model: string | null;
  status: string;
  job_id: string;
  metrics: Metrics | null;
  by_family?: Record<string, Metrics>;
  by_env?: Record<string, Metrics>;
  by_difficulty?: Record<string, Metrics>;
  by_role?: Record<string, RoleMetrics>;
  n_tasks: number;
  created_at: number;
  finished_at: number | null;
  agent_name?: string;
  agent_version?: string;
}
export interface RolloutRecord {
  rollout_id: string;
  id?: string;
  task_id: string;
  env_name: string;
  family: string;
  difficulty: string;
  trial: number;
  status: string;
  n_steps: number;
  n_tool_calls: number;
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  task_success: number;
  tool_selection_f1: number;
  step_efficiency: number;
  context_preservation: number;
  tool_error_free: number;
  judge_score: number | null;
  rationale: string;
  called_tools: string[];
  trace_id: string | null;
}
export interface EvalDetail extends EvalRun {
  agent: Agent | null;
  job: Job | null;
  rollouts: RolloutRecord[];
}
export interface Comparison {
  baseline: string;
  candidate: string;
  comparison: Record<string, { baseline: number; candidate: number; delta: number; p_value: number; ci_candidate: [number, number] }>;
}

export interface TrainingHistoryItem {
  iteration: number;
  version: string;
  collect_success: number | null;
  eval: Metrics | null;
  seconds: number;
  agent_id: string;
}
export interface TrainingRun {
  id: string;
  agent_id: string;
  algorithm: string;
  params: Record<string, unknown>;
  status: string;
  job_id: string;
  result: Record<string, unknown> | null;
  output_agent_id: string | null;
  history: TrainingHistoryItem[];
  created_at: number;
  finished_at: number | null;
  agent_name?: string;
  input_version?: string;
  output_version?: string | null;
  job?: Job | null;
}

export interface TraceSummary {
  id: string;
  name: string;
  service: string;
  start_time: number;
  end_time: number;
  status: string;
  n_spans: number;
  attributes: Record<string, unknown>;
  rollout_id: string | null;
  created_at: number;
  duration_ms: number;
}
export interface Span {
  id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  kind: string;
  start_time: number;
  end_time: number;
  status: string;
  attributes: Record<string, unknown>;
  events: unknown[];
}
export interface Score {
  id: string;
  trace_id: string | null;
  rollout_id: string | null;
  name: string;
  value: number;
  source: string;
  rationale: string;
  step_index: number | null;
  created_at: number;
}
export interface TraceDetail extends Omit<TraceSummary, "duration_ms"> {
  spans: Span[];
  scores: Score[];
}

export interface ChatMessage {
  role: string;
  content: string;
  tool_calls: { id: string; name: string; arguments: Record<string, unknown> }[];
  tool_call_id: string | null;
  name: string | null;
}
export interface RolloutStep {
  index: number;
  prompt_messages: ChatMessage[];
  response: ChatMessage;
  tool_results: { call_id: string; name: string; output: unknown; error: string | null; latency_ms: number }[];
  usage: { prompt_tokens: number; completion_tokens: number };
  latency_ms: number;
  exposed_tools: string[];
  reward: number | null;
}
export interface Reward {
  value: number;
  source: string;
  name: string;
  step_index: number | null;
  rationale: string | null;
  metadata: Record<string, unknown>;
}
export interface Rollout {
  id: string;
  agent_id: string;
  task_id: string;
  env_name: string;
  status: string;
  task_success: number | null;
  total_reward: number | null;
  record: RolloutRecord;
  trace_id: string | null;
  eval_run_id: string | null;
  training_run_id: string | null;
  created_at: number;
}
export interface RolloutDetail extends Rollout {
  payload: {
    agent_id: string;
    agent_version: string;
    steps: RolloutStep[];
    final_answer: string;
    status: string;
    rewards: Reward[];
    metadata: Record<string, unknown>;
  } | null;
  scores: Score[];
}

export interface Deployment {
  id: string;
  agent_name: string;
  agent_id: string;
  baseline_agent_id: string | null;
  eval_run_id: string | null;
  policy: Record<string, unknown>;
  decision: {
    passed: boolean;
    reasons: string[];
    checks: Record<string, { value?: number; threshold?: number; delta?: number; p_value?: number; ok: boolean }>;
    comparison?: Record<string, unknown>;
  };
  promoted: boolean;
  active: boolean;
  created_at: number;
  agent_version?: string;
  baseline_version?: string | null;
}

export interface Experiment {
  id: string;
  name: string;
  variants: { name: string; agent_id: string; weight: number }[];
  status: string;
  created_at: number;
  n_outcomes: number;
  results: Record<string, { n: number; mean: number; ci: [number, number] } | number> & { p_value?: number; delta?: number };
}

export interface Dataset {
  id: string;
  name: string;
  suite: string | null;
  split: string;
  created_at: number;
  n_tasks: number;
  tasks?: TaskSpec[];
}
export interface TaskSpec {
  id: string;
  env_name: string;
  instruction: string;
  expected: Record<string, unknown>;
  user_script: string[];
  difficulty: string;
  tags: string[];
  max_steps: number;
}
export interface Environment {
  name: string;
  description: string;
  n_tools: number;
  tools: string[];
}
export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  tags: string[];
  source: string;
}
export interface TimeseriesPoint {
  ts: number;
  value: number;
  agent_id: string;
  version: string;
  tags: Record<string, unknown>;
}
export interface Overview {
  project: { id: string; name: string };
  n_agents: number;
  n_traces: number;
  n_rollouts: number;
  rollout_success_rate: number | null;
  n_datasets: number;
  n_eval_runs: number;
  n_training_runs: number;
  agents: Record<
    string,
    {
      versions: number;
      deployed_version: string | null;
      deployed_agent_id: string | null;
      latest_eval: EvalRun | null;
      first_task_success: number | null;
      latest_task_success: number | null;
      [k: string]: unknown;
    }
  >;
  recent_jobs: Job[];
  [k: string]: unknown;
}
export interface Paged<T> {
  total: number;
  items: T[];
}

// ---- auth / org / audit -------------------------------------------------------
export interface Session {
  token?: string;
  apiKey?: string;
  org?: string;
}
const SESSION_KEY = "saphire.session";
let sessionCache: Session | null = null;

export function getSession(): Session {
  if (sessionCache) return sessionCache;
  try {
    if (typeof window !== "undefined") {
      const raw = window.localStorage.getItem(SESSION_KEY);
      sessionCache = raw ? (JSON.parse(raw) as Session) : {};
    } else sessionCache = {};
  } catch {
    sessionCache = {};
  }
  return sessionCache!;
}
export function setSession(patch: Session) {
  const next: Session = { ...getSession(), ...patch };
  for (const k of Object.keys(next) as (keyof Session)[]) if (!next[k]) delete next[k];
  sessionCache = next;
  try {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(next));
  } catch {}
}
export function clearSession() {
  sessionCache = {};
  try {
    window.localStorage.removeItem(SESSION_KEY);
  } catch {}
}
/** Headers used for every API request (Bearer token wins over API key). */
export function authHeaders(): Record<string, string> {
  const s = getSession();
  const h: Record<string, string> = s.token ? { authorization: `Bearer ${s.token}` } : { "x-api-key": s.apiKey || API_KEY };
  if (s.org) h["x-org"] = s.org;
  return h;
}

export interface AuthConfig {
  oidc: boolean;
  dev_login: boolean;
  login_url: string | null;
}
export interface Me {
  actor_type: "root" | "user" | "api_key" | string;
  label: string;
  role: string;
  org: string | null;
  user: { id: string; email: string; name: string; is_superadmin: boolean } | null;
  orgs: { slug: string; name: string; role: string }[];
  permissions: string[];
}
export interface Org {
  id: string;
  name: string;
  slug: string;
  plan: string;
  quotas: Record<string, number>;
  settings: { allowed_email_domains?: string[]; retention_days?: number; default_role?: string; [k: string]: unknown };
  created_at: number;
  effective_quotas: Record<string, number>;
  n_members: number;
  n_projects: number;
  n_api_keys: number;
  you?: { actor_type: string; label: string; role: string };
}
export interface Member {
  membership_id: string;
  user_id: string;
  email: string;
  name: string;
  role: string;
  sso_provider: string | null;
  last_login_at: number | null;
  created_at: number;
}
export interface ApiKey {
  id: string;
  org_id: string;
  name: string;
  prefix: string;
  role: string;
  created_by: string;
  created_at: number;
  last_used_at: number | null;
  expires_at: number | null;
  revoked_at: number | null;
  key?: string; // only on create
}
export interface Usage {
  days: number;
  totals: Record<string, number>;
  daily: Record<string, Record<string, number>>;
  quotas: Record<string, number>;
  plan: string;
}
export interface AuditItem {
  id: string;
  org_id: string;
  actor_type: string;
  actor_id: string;
  actor_label: string;
  action: string;
  resource_type: string;
  resource_id: string;
  method: string;
  path: string;
  status_code: number;
  ip: string;
  details: Record<string, unknown>;
  created_at: number;
}

// ---- intelligence / attribution / webhooks / signals ---------------------------
export interface DriftAlert {
  type: string;
  metric?: string;
  family?: string;
  message?: string;
  delta?: number;
  p_value?: number;
  ratio?: number;
  value?: number;
  severity?: string;
  [k: string]: unknown;
}
export interface FailureCluster {
  kind: string;
  role: string;
  tool: string;
  family: string;
  count: number;
  rate: number;
  baseline_rate: number | null;
  trend: number | null;
  examples: string[];
  tasks: string[];
}
export interface Recommendation {
  priority: number;
  title: string;
  why: string;
  action: { kind: string; algorithm?: string; params?: Record<string, unknown>; what?: string; [k: string]: unknown };
}
export interface IntelligenceReport {
  id: string;
  project_id: string;
  agent_name: string | null;
  recent_hours: number;
  baseline_hours: number;
  n_recent: number;
  n_baseline: number;
  healthy: boolean;
  drift: {
    n_baseline: number;
    n_recent: number;
    metrics: Record<string, { baseline: number; recent: number; delta: number; p_value: number }>;
    alerts: DriftAlert[];
    note?: string;
  };
  failures: { n_rollouts: number; n_failed: number; failure_rate: number; clusters: FailureCluster[]; by_kind?: Record<string, number>; by_role?: Record<string, number> };
  coverage: {
    production_patterns?: number;
    covered_patterns?: number;
    uncovered_patterns: number;
    uncovered_traffic_share: number;
    top_uncovered?: { tools: string[]; count: number }[];
    tools_never_evaluated?: string[];
    minable_tasks?: number;
  };
  recommendations: Recommendation[];
  job_id: string | null;
  created_at: number;
}
export type IntelligenceReportSummary = Pick<IntelligenceReport, "id" | "agent_name" | "recent_hours" | "baseline_hours" | "n_recent" | "n_baseline" | "healthy" | "created_at" | "recommendations"> & {
  n_alerts: number;
  n_recommendations: number;
};
export interface ApplyResult {
  kind: string;
  job_id?: string;
  training_run_id?: string;
  algorithm?: string;
  dataset_id?: string;
  n_tasks?: number;
  [k: string]: unknown;
}
export interface MineResult {
  dataset_id: string;
  n_tasks: number;
  from_rollouts: number;
  families: string[];
}

export interface AttributionResult {
  agent_id: string;
  agent_version: string;
  dataset_id: string;
  n_tasks: number;
  ablation?: {
    baseline: number;
    roles: Record<string, { upgraded: number; headroom: number; degraded: number; criticality: number }>;
    reference_model?: string | null;
    degraded_model?: string | null;
    all_upgraded?: number;
    total_headroom?: number;
  };
  shapley?: {
    baseline: number;
    all_upgraded: number;
    shapley: Record<string, number>;
    share?: Record<string, number>;
    n_permutations: number;
    n_evaluations: number;
    reference_model?: string | null;
  };
  recommendation?: { optimize_roles: string[]; reason: string };
}
export interface AttributionJob extends Omit<Job, "result"> {
  result: AttributionResult | null;
}
export interface EvalAttribution {
  eval_run_id: string;
  credit: {
    n_rollouts: number;
    n_failed: number;
    roles: Record<
      string,
      {
        rollouts: number;
        fault_share: number;
        first_faults: number;
        fault_kinds: Record<string, number>;
        success_when_clean: number | null;
        success_when_faulty: number | null;
        advantage: number | null;
        mistake_rate: number;
      }
    >;
  };
  blame: { role: string; step_index: number | null; tool: string | null; kind: string; detail: string; success: boolean; rollout_id: string; task_id: string }[];
  by_kind?: Record<string, number>;
}

export interface WebhookDelivery {
  id: string;
  webhook_id: string;
  event: string;
  payload: Record<string, unknown>;
  status_code: number;
  error: string;
  attempts: number;
  created_at: number;
}
export interface Webhook {
  id: string;
  org_id: string;
  url: string;
  events: string[];
  description: string;
  active?: boolean;
  created_at: number;
  last_delivery: WebhookDelivery | null;
  secret?: string; // only on create
  note?: string;
}

export interface Calibration {
  n: number;
  judge: string;
  paired_rollouts: number;
  note?: string;
  agreement?: number;
  kappa?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  threshold?: number;
  best_threshold?: number;
  best_agreement?: number;
  reliability?: Record<string, { n: number; human_positive_rate: number }>;
}
export type ToolStats = Record<string, { calls: number; error_rate: number; success_rate: number; latency_ms: number; confused_with: Record<string, number> }>;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function url(path: string, params?: Record<string, string | number | boolean | undefined | null>) {
  const u = new URL(API_BASE.replace(/\/$/, "") + path);
  if (!("project" in (params || {}))) u.searchParams.set("project", PROJECT);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, String(v));
  }
  return u.toString();
}

export async function api<T>(
  path: string,
  opts: { method?: string; body?: unknown; params?: Record<string, string | number | boolean | undefined | null> } = {},
): Promise<T> {
  const res = await fetch(url(path, opts.params), {
    method: opts.method || "GET",
    headers: { ...authHeaders(), ...(opts.body !== undefined ? { "content-type": "application/json" } : {}) },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {}
    if (res.status === 401 && typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
      // Session is invalid / missing: drop a stale token and go to the login page.
      if (getSession().token) setSession({ token: undefined });
      window.location.href = "/login/";
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// SWR fetcher: key is a path (optionally with query string already appended)
export const fetcher = <T,>(key: string) => api<T>(key);

/** Build an SWR key with query params. */
export function q(path: string, params: Record<string, string | number | boolean | undefined | null>) {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  const s = sp.toString();
  return s ? `${path}?${s}` : path;
}

export const Api = {
  overview: () => api<Overview>("/v1/metrics/overview"),
  agents: (name?: string) => api<Agent[]>("/v1/agents", { params: { name } }),
  agent: (id: string) => api<Agent>(`/v1/agents/${id}`),
  promoteAgent: (id: string) => api<Agent>(`/v1/agents/${id}/promote`, { method: "POST" }),
  createEval: (body: { agent_id: string; dataset_id: string; k: number; judge_model: string | null; gate: boolean; auto_promote: boolean }) =>
    api<EvalRun>("/v1/evals", { method: "POST", body }),
  compareEvals: (id: string, other: string) => api<Comparison>(`/v1/evals/${id}/compare/${other}`),
  createTraining: (body: { agent_id: string; algorithm: string; params: Record<string, unknown> }) =>
    api<TrainingRun>("/v1/training", { method: "POST", body }),
  promoteDeployment: (id: string) => api<Deployment>(`/v1/deployments/${id}/promote`, { method: "POST" }),
  createExperiment: (body: { name: string; variants: { name: string; agent_id: string; weight: number }[] }) =>
    api<Experiment>("/v1/experiments", { method: "POST", body }),
  stopExperiment: (id: string) => api<Experiment>(`/v1/experiments/${id}/stop`, { method: "POST" }),
  cancelJob: (id: string) => api<Job>(`/v1/jobs/${id}/cancel`, { method: "POST" }),
  // auth / org
  authConfig: () => api<AuthConfig>("/v1/auth/config", { params: { project: undefined } }),
  devLogin: (body: { email: string; name?: string; org?: string | null; role?: string }) => api<{ token: string }>("/v1/auth/dev-login", { method: "POST", body, params: { project: undefined } }),
  me: () => api<Me>("/v1/auth/me"),
  orgs: () => api<Org[]>("/v1/orgs"),
  currentOrg: () => api<Org>("/v1/orgs/current"),
  patchOrg: (body: { name?: string; plan?: string; quotas?: Record<string, number>; settings?: Record<string, unknown> }) => api<Org>("/v1/orgs/current", { method: "PATCH", body }),
  addMember: (body: { email: string; role: string; name?: string }) => api<Member>("/v1/orgs/current/members", { method: "POST", body }),
  removeMember: (id: string) => api<unknown>(`/v1/orgs/current/members/${id}`, { method: "DELETE" }),
  createKey: (body: { name: string; role: string; expires_in_days: number | null }) => api<ApiKey>("/v1/orgs/current/keys", { method: "POST", body }),
  revokeKey: (id: string) => api<unknown>(`/v1/orgs/current/keys/${id}`, { method: "DELETE" }),
  // intelligence
  runIntelligence: (body: { agent_name: string | null; recent_hours: number; baseline_hours: number; production_only: boolean }) =>
    api<IntelligenceReport>("/v1/intelligence/run", { method: "POST", body: { ...body, sync: true } }),
  applyRecommendation: (body: { recommendation: Recommendation; agent_id: string | null; train_dataset_id: string | null; eval_dataset_id: string | null }) =>
    api<ApplyResult>("/v1/intelligence/apply", { method: "POST", body }),
  mineTasks: (body: { name: string; since_hours: number; production_only: boolean }) => api<MineResult>("/v1/intelligence/mine-tasks", { method: "POST", body }),
  // attribution
  createAttribution: (body: { agent_id: string; dataset_id: string; reference_model: string | null; degraded_model: string | null; shapley: boolean; n_permutations: number }) =>
    api<AttributionJob>("/v1/attribution", { method: "POST", body }),
  job: (id: string) => api<Job>(`/v1/jobs/${id}`),
  // webhooks
  createWebhook: (body: { url: string; events: string[]; description: string; secret?: string }) => api<Webhook>("/v1/orgs/current/webhooks", { method: "POST", body }),
  deleteWebhook: (id: string) => api<unknown>(`/v1/orgs/current/webhooks/${id}`, { method: "DELETE" }),
  testWebhook: (id: string) => api<WebhookDelivery[]>(`/v1/orgs/current/webhooks/${id}/test`, { method: "POST" }),
};
