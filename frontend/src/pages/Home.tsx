import { Trash2, TrendingUp, TrendingDown, Activity } from "lucide-react";
import { useState } from "react";
import type { RunMeta } from "../types";

interface Props {
  runs: RunMeta[];
  onSelectRun: (tag: string) => void;
  onDeleteRun: (tag: string) => void;
  onNewRun: () => void;
}

interface ConfirmState { tag: string }

export function Home({ runs, onSelectRun, onDeleteRun, onNewRun }: Props) {
  const [confirmDelete, setConfirmDelete] = useState<ConfirmState | null>(null);

  function accDelta(r: RunMeta) {
    if (r.acc_static == null || r.acc_adaptive == null) return null;
    return (r.acc_adaptive - r.acc_static) * 100;
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Header */}
      <div className="flex items-end justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-[#f5f5f5] tracking-tight">Run History</h1>
          <p className="text-sm text-[#6b6b7a] mt-1 font-mono">{runs.length} result{runs.length !== 1 ? "s" : ""} available</p>
        </div>
        <button onClick={onNewRun} className="btn-primary">
          + New Run
        </button>
      </div>

      {runs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <Activity size={40} className="text-[#2a2a2d]" />
          <p className="text-[#6b6b7a] text-sm">No runs yet.</p>
          <button onClick={onNewRun} className="btn-primary">Configure &amp; Launch &rarr;</button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {runs.map((r) => {
            const delta = accDelta(r);
            return (
              <div
                key={r.tag}
                className="card-hover group relative flex flex-col gap-3"
                onClick={() => onSelectRun(r.tag)}
              >
                {/* Delete button */}
                <button
                  className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity btn-danger p-1.5 rounded-lg z-10"
                  onClick={(e) => { e.stopPropagation(); setConfirmDelete({ tag: r.tag }); }}
                  title="Delete run"
                >
                  <Trash2 size={14} />
                </button>

                {/* Tag */}
                <div>
                  <p className="font-mono text-xs text-[#f5f5f5] pr-8 truncate">{r.tag}</p>
                  <p className="font-mono text-[10px] text-[#6b6b7a] mt-0.5">
                    {r.date_min?.slice(0,10)} &rarr; {r.date_max?.slice(0,10)}
                  </p>
                </div>

                <div className="divider" />

                {/* Stats */}
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <p className="label mb-1">Obs</p>
                    <p className="font-mono text-sm text-[#f5f5f5]">
                      {r.n_obs != null ? r.n_obs.toLocaleString() : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="label mb-1">Drift Events</p>
                    <p className="font-mono text-sm" style={{ color: r.n_drift_events ? "#fb923c" : "#6b6b7a" }}>
                      {r.n_drift_events ?? "—"}
                    </p>
                  </div>
                  <div>
                    <p className="label mb-1">Acc &Delta;</p>
                    <p className="font-mono text-sm flex items-center gap-1">
                      {delta == null ? (
                        <span className="text-[#6b6b7a]">—</span>
                      ) : delta > 0 ? (
                        <><TrendingUp size={12} className="text-[#4ade80]" /><span className="text-[#4ade80]">+{delta.toFixed(1)}pp</span></>
                      ) : delta < 0 ? (
                        <><TrendingDown size={12} className="text-[#f87171]" /><span className="text-[#f87171]">{delta.toFixed(1)}pp</span></>
                      ) : (
                        <span className="text-[#6b6b7a]">0pp</span>
                      )}
                    </p>
                  </div>
                </div>

                {/* Accuracy bars */}
                {r.acc_static != null && r.acc_adaptive != null && (
                  <div className="space-y-1.5">
                    {[
                      { label: "Static",   val: r.acc_static,   color: "#6b6b7a" },
                      { label: "Adaptive", val: r.acc_adaptive, color: "#d4a853" },
                    ].map(({ label, val, color }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="font-mono text-[10px] w-14" style={{ color }}>{label}</span>
                        <div className="flex-1 h-1 bg-[#2a2a2d] rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${val * 100}%`, background: color }} />
                        </div>
                        <span className="font-mono text-[10px] w-10 text-right" style={{ color }}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Delete confirmation modal */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-[#1e1e21] border border-[#2a2a2d] rounded-2xl p-6 max-w-sm w-full shadow-2xl">
            <h2 className="text-[#f5f5f5] font-semibold mb-2">Delete run?</h2>
            <p className="font-mono text-xs text-[#6b6b7a] mb-1 break-all">{confirmDelete.tag}</p>
            <p className="text-sm text-[#6b6b7a] mb-6">This will permanently delete all result files. This cannot be undone.</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setConfirmDelete(null)} className="btn-ghost">Cancel</button>
              <button
                onClick={() => { onDeleteRun(confirmDelete.tag); setConfirmDelete(null); }}
                className="btn-danger border border-[#f87171]/30 px-4 py-2 rounded-lg text-sm"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
