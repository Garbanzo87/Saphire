# Saphire Console

Web dashboard for the Saphire agent stack: traces, rollouts, evaluations, training runs, gated deployments, experiments, datasets, environments and jobs.

Next.js 14 (app router, TypeScript, Tailwind, recharts, SWR). All data is fetched **client-side** from the Saphire API, so the app is exported as a fully static site (`output: 'export'`) and can be served by any static file server.

## Development

```bash
npm install
npm run dev          # http://localhost:3000
```

The Saphire API must be running (default `http://localhost:8000`) and allow CORS from the dashboard origin.

## Configuration

Set via environment variables (or a `.env.local`, see `.env.example`). Because they are `NEXT_PUBLIC_*` they are inlined **at build time**.

| Variable                     | Default                 | Purpose                                   |
| ---------------------------- | ----------------------- | ----------------------------------------- |
| `NEXT_PUBLIC_SAPHIRE_API`     | `http://localhost:8000` | Base URL of the Saphire API               |
| `NEXT_PUBLIC_SAPHIRE_API_KEY` | `dev-key`               | Sent as the `x-api-key` header            |
| `NEXT_PUBLIC_SAPHIRE_PROJECT` | `default`               | Project name passed as `?project=` on requests |

## Production build

```bash
npm run build        # writes the static site to ./out
npx serve out        # or any static server
```

Detail pages use query parameters (`/traces/detail?id=…`, `/evals/detail?id=…`, `/training/detail?id=…`, `/rollouts/detail?id=…`, `/agents/detail?id=…`, `/jobs?id=…`) so no dynamic routes are needed for the static export.

## Docker

```bash
docker build \
  --build-arg NEXT_PUBLIC_SAPHIRE_API=https://api.example.com \
  --build-arg NEXT_PUBLIC_SAPHIRE_API_KEY=my-key \
  --build-arg NEXT_PUBLIC_SAPHIRE_PROJECT=default \
  -t saphire-console .
docker run -p 3000:3000 saphire-console
```

The image builds the static export and serves it with nginx on port 3000.

## Pages

| Route           | Contents                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------- |
| `/`             | KPI tiles, per-agent cards (deployed version, first vs latest task_success), metric timeseries, recent jobs |
| `/agents`       | Agent versions with Promote; detail shows system prompt, config, lineage, eval runs          |
| `/traces`       | Paginated traces; detail shows a span waterfall, span attributes and attached scores         |
| `/rollouts`     | Filterable rollouts; detail reconstructs the conversation, rewards and verifier checks       |
| `/evals`        | Eval runs, "New evaluation" form; detail has metrics, breakdown charts, gate result, per-rollout table and compare-with |
| `/training`     | Training runs, "New training run" form; detail charts per-iteration metrics, versions, logs  |
| `/deployments`  | Currently deployed agents and gate decisions with checks and Promote                         |
| `/experiments`  | A/B experiments with variant results and a create form                                       |
| `/datasets`, `/environments` | Dataset and task preview; environment tool lists and specs                      |
| `/jobs`         | Job table with progress, errors and log expander                                             |

List pages refresh every 5 s; eval/training detail pages poll while the run is pending/running.
