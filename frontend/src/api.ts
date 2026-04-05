const BASE = "/api";

async function get<T>(url: string): Promise<T> {
  const r = await fetch(BASE + url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const api = {
  runs:         ()        => get<import("./types").RunMeta[]>("/runs"),
  daily:        (tag: string) => get<import("./types").DailyRow[]>(`/runs/${encodeURIComponent(tag)}/daily`),
  equity:       (tag: string) => get<import("./types").EquityRow[]>(`/runs/${encodeURIComponent(tag)}/equity`),
  driftEvents:  (tag: string) => get<import("./types").DriftEvent[]>(`/runs/${encodeURIComponent(tag)}/drift_events`),
  evaluate:     (tag: string) => get<import("./types").EvalResult>(`/runs/${encodeURIComponent(tag)}/evaluate`),
  correlations: (tag: string) => get<import("./types").CorrelationData>(`/runs/${encodeURIComponent(tag)}/correlations`),
  rocPr:        (tag: string) => get<import("./types").RocPrData>(`/runs/${encodeURIComponent(tag)}/roc_pr`),
  deleteRun:    (tag: string) => fetch(`${BASE}/runs/${encodeURIComponent(tag)}`, { method: "DELETE" }).then(r => r.json()),
  createRun:    (cfg: Record<string, unknown>) => fetch(`${BASE}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg) }).then(r => r.json()) as Promise<{ run_id: string }>,
};
