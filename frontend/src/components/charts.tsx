"use client";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export const SERIES_COLORS = ["#5b9cff", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#22d3ee"];
const axis = { stroke: "#8b94a7", fontSize: 11 };
const tooltipStyle = { contentStyle: { background: "#12161f", border: "1px solid #232a38", borderRadius: 6, fontSize: 12 }, labelStyle: { color: "#8b94a7" } };

export function MetricLineChart({
  data,
  xKey,
  series,
  height = 260,
  percent = true,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  series: { key: string; label?: string }[];
  height?: number;
  percent?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="#232a38" strokeDasharray="3 3" />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} domain={percent ? [0, 1] : ["auto", "auto"]} tickFormatter={(v) => (percent ? `${Math.round(v * 100)}%` : v)} />
        <Tooltip {...tooltipStyle} formatter={(v) => (percent && typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v))} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s, i) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label || s.key} stroke={SERIES_COLORS[i % SERIES_COLORS.length]} strokeWidth={2} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function MetricBarChart({
  data,
  xKey,
  series,
  height = 220,
  percent = true,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  series: { key: string; label?: string }[];
  height?: number;
  percent?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="#232a38" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} domain={percent ? [0, 1] : ["auto", "auto"]} tickFormatter={(v) => (percent ? `${Math.round(v * 100)}%` : v)} />
        <Tooltip {...tooltipStyle} cursor={{ fill: "#181d28" }} formatter={(v) => (percent && typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v))} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s, i) => (
          <Bar key={s.key} dataKey={s.key} name={s.label || s.key} fill={SERIES_COLORS[i % SERIES_COLORS.length]} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
