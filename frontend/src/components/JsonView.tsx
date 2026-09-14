"use client";
import { useState } from "react";

function Node({ k, v, depth }: { k: string | null; v: unknown; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  const isObj = v !== null && typeof v === "object";
  const key = k !== null ? <span className="text-sky-300">{JSON.stringify(k)}</span> : null;
  if (!isObj) {
    let cls = "text-emerald-300";
    if (typeof v === "number") cls = "text-amber-300";
    else if (typeof v === "boolean") cls = "text-violet-300";
    else if (v === null) cls = "text-zinc-500";
    const text = typeof v === "string" ? JSON.stringify(v) : String(v);
    return (
      <div className="whitespace-pre-wrap break-words">
        {key}
        {key && ": "}
        <span className={cls}>{text}</span>
      </div>
    );
  }
  const entries = Array.isArray(v) ? v.map((x, i) => [String(i), x] as const) : Object.entries(v as Record<string, unknown>);
  const [ob, cb] = Array.isArray(v) ? ["[", "]"] : ["{", "}"];
  if (entries.length === 0)
    return (
      <div>
        {key}
        {key && ": "}
        {ob}
        {cb}
      </div>
    );
  return (
    <div>
      <button type="button" onClick={() => setOpen(!open)} className="text-left hover:text-white">
        <span className="mr-1 inline-block w-3 text-zinc-500">{open ? "▾" : "▸"}</span>
        {key}
        {key && ": "}
        {ob}
        {!open && <span className="text-zinc-500"> {entries.length} items {cb}</span>}
      </button>
      {open && (
        <div className="ml-4 border-l border-[var(--border)] pl-3">
          {entries.map(([ck, cv]) => (
            <Node key={ck} k={Array.isArray(v) ? null : ck} v={cv} depth={depth + 1} />
          ))}
          <div>{cb}</div>
        </div>
      )}
    </div>
  );
}

/** Collapsible, syntax-tinted JSON tree. Strings that contain JSON are shown as-is (use tryJson upstream). */
export function JsonView({ value, className = "" }: { value: unknown; className?: string }) {
  return (
    <div className={`mono max-h-[32rem] overflow-auto rounded-md bg-[var(--bg)] p-3 text-[12px] leading-5 ${className}`}>
      <Node k={null} v={value} depth={0} />
    </div>
  );
}
