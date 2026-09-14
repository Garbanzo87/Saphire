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
  metadata?: Record<string, unknown>;
  [k: string]: unknown;
}
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
    headers: { "x-api-key": API_KEY, ...(opts.body !== undefined ? { "content-type": "application/json" } : {}) },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {}
    throw new ApiError(res.status, msg);
  }
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
};
