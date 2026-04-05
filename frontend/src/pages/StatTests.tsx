import { useEffect, useState } from "react";
import { api } from "../api";
import type { EvalResult, RunMeta } from "../types";

interface Props { run: RunMeta; }

function pColor(p: number | null | undefined, threshold: number) {
  if (p == null) return "#6b6b7a";
  return p < threshold ? "#4ade80" : "#f87171";
}

function SigLabel({ sig, pval, threshold = 0.05 }: { sig?: boolean | null; pval?: number | null; threshold?: number }) {
  if (pval != null) {
    const significant = pval < threshold;
    return (
      <span className={`pill ${significant ? "pill-green" : "pill-slate"}`}>
        p = {pval < 0.0001 ? pval.toExponential(2) : pval.toFixed(4)}
        {" "}&middot; {significant ? "sig" : "not sig"}
      </span>
    );
  }
  if (sig == null) return <span className="pill pill-slate">n/a</span>;
  return <span className={sig ? "pill-green pill" : "pill pill-slate"}>{sig ? "sig" : "not sig"}</span>;
}

function CIBar({ lo, hi, pt, color }: { lo: number; hi: number; pt: number; color: string }) {
  const MIN = Math.min(-1, lo - 0.05);
  const MAX = Math.max(1, hi + 0.05);
  const range = MAX - MIN;
  const pct = (v: number) => `${((v - MIN) / range) * 100}%`;
  const excludesZero = lo > 0 || hi < 0;

  return (
    <div className="space-y-1">
      <div className="relative h-5 bg-[#0f0f0f] rounded overflow-hidden border border-[#2a2a2d]">
        <div className="absolute top-0 bottom-0 w-px bg-[#3f3f46]" style={{ left: pct(0) }} />
        <div
          className="absolute top-1 bottom-1 rounded opacity-20"
          style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})`, background: color }}
        />
        <div
          className="absolute top-0.5 bottom-0.5 w-0.5 rounded"
          style={{ left: pct(pt), background: color }}
        />
      </div>
      <p className="font-mono text-[10px] text-[#6b6b7a]">
        {pt > 0 ? "+" : ""}{pt.toFixed(4)} &middot; 95% CI [{lo.toFixed(3)}, {hi.toFixed(3)}]
        {" "}&middot; {excludesZero
          ? lo > 0 ? "CI above 0 — significant improvement" : "CI below 0 — significant decline"
          : "CI straddles 0 — not significant"}
      </p>
    </div>
  );
}

export function StatTests({ run }: Props) {
  const [result, setResult] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function load() {
    setLoading(true);
    setError(null);
    api.evaluate(run.tag)
      .then((r) => { setResult(r); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }

  useEffect(() => { load(); }, [run.tag]);

  const GOLD = "#d4a853";
  const bonf = result?.alpha_bonferroni ?? 0.01;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-[#f5f5f5] tracking-tight">Statistical Significance</h1>
          <p className="font-mono text-[11px] text-[#6b6b7a] mt-1">
            5 tests &middot; Bonferroni &alpha; = {bonf.toFixed(4)} &middot; bootstrap n = 3000
          </p>
        </div>
        <button onClick={load} disabled={loading} className="btn-ghost">
          {loading ? "Running…" : "Re-run"}
        </button>
      </div>

      {error && <div className="card border-[#f87171]/30 font-mono text-xs text-[#f87171]">Error: {error}</div>}
      {loading && !result && <div className="font-mono text-sm text-[#6b6b7a] text-center py-12">Running statistical tests…</div>}

      {result && (
        <>
          {/* Summary row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              {
                label: "McNemar p",
                val: result.mcnemar_p.toFixed(4),
                sig: result.mcnemar_sig_bonf,
                pval: result.mcnemar_p,
                threshold: bonf,
              },
              {
                label: "OLS β (drift)",
                val: `${result.ols_beta > 0 ? "+" : ""}${result.ols_beta.toFixed(5)}`,
                sig: result.ols_sig_bonf,
                pval: result.ols_p,
                threshold: bonf,
              },
              {
                label: "Sharpe Δ (LO)",
                val: `${result.sharpe_long_pt > 0 ? "+" : ""}${result.sharpe_long_pt.toFixed(4)}`,
                sig: result.sharpe_long_sig,
                pval: null,
                threshold: bonf,
              },
              {
                label: "ΔAUC",
                val: result.auc_pt != null ? `${result.auc_pt > 0 ? "+" : ""}${result.auc_pt.toFixed(4)}` : "—",
                sig: result.auc_sig,
                pval: null,
                threshold: bonf,
              },
            ].map(({ label, val, sig, pval, threshold }) => (
              <div key={label} className="card-sm flex flex-col gap-2">
                <span className="label">{label}</span>
                <span
                  className="font-mono text-xl"
                  style={{ color: sig ? "#4ade80" : sig === false ? "#f87171" : "#f5f5f5", fontVariantNumeric: "tabular-nums" }}
                >
                  {val}
                </span>
                <SigLabel sig={sig} pval={pval} threshold={threshold} />
              </div>
            ))}
          </div>

          {/* McNemar */}
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <p className="font-semibold text-[#f5f5f5] text-sm">McNemar Test</p>
              <SigLabel sig={result.mcnemar_sig_bonf} pval={result.mcnemar_p} threshold={bonf} />
            </div>
            <p className="font-mono text-[11px] text-[#6b6b7a] leading-relaxed">
              H&#8320;: static and adaptive classifiers have equal error rates. Continuity-corrected McNemar &chi;².
            </p>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-[#0f0f0f] rounded-lg px-3 py-2.5">
                <p className="label mb-1">p-value (raw)</p>
                <p className="font-mono" style={{ color: pColor(result.mcnemar_p, 0.05) }}>
                  {result.mcnemar_p.toFixed(4)}
                </p>
              </div>
              <div className="bg-[#0f0f0f] rounded-lg px-3 py-2.5">
                <p className="label mb-1">Bonferroni (&alpha; = {bonf.toFixed(4)})</p>
                <div className="mt-1">
                  <SigLabel sig={result.mcnemar_sig_bonf} pval={result.mcnemar_p} threshold={bonf} />
                </div>
              </div>
            </div>
          </div>

          {/* OLS */}
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <p className="font-semibold text-[#f5f5f5] text-sm">OLS Regression on Drift Days</p>
              <SigLabel sig={result.ols_sig_bonf} pval={result.ols_p} threshold={bonf} />
            </div>
            <p className="font-mono text-[11px] text-[#6b6b7a] leading-relaxed">
              logloss_adaptive = &alpha; + &beta; &middot; drift_event + &epsilon; (HC1 robust SE)
            </p>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "β (drift coeff)", val: `${result.ols_beta > 0 ? "+" : ""}${result.ols_beta.toFixed(5)}` },
                { label: "p-value", val: result.ols_p.toFixed(4) },
                { label: "Bonf. corrected", val: result.ols_sig_bonf ? "significant" : "not sig" },
              ].map(({ label, val }) => (
                <div key={label} className="bg-[#0f0f0f] rounded-lg px-3 py-2.5">
                  <p className="label mb-1">{label}</p>
                  <p className="font-mono text-sm text-[#f5f5f5]">{val}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Sharpe Bootstrap */}
          <div className="card space-y-3">
            <p className="font-semibold text-[#f5f5f5] text-sm">Bootstrap Sharpe Difference (3000 resamples)</p>
            <p className="font-mono text-[11px] text-[#6b6b7a] leading-relaxed">
              Non-parametric bootstrap CI for Sharpe(adaptive) &minus; Sharpe(static). CI excluding 0 &rarr; significant.
            </p>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs text-[#6b6b7a]">Long-Only Strategy</span>
                  <span className={`font-mono text-sm ${result.sharpe_long_pt >= 0 ? "text-[#4ade80]" : "text-[#f87171]"}`}>
                    {result.sharpe_long_pt > 0 ? "+" : ""}{result.sharpe_long_pt.toFixed(4)}
                  </span>
                </div>
                {result.sharpe_long_lo != null && result.sharpe_long_hi != null ? (
                  <CIBar lo={result.sharpe_long_lo} hi={result.sharpe_long_hi} pt={result.sharpe_long_pt} color={GOLD} />
                ) : (
                  <p className="font-mono text-[10px] text-[#6b6b7a]">
                    {result.sharpe_long_sig ? "CI excludes 0 — significant" : "CI includes 0 — not significant"}
                  </p>
                )}
              </div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-[#6b6b7a]">Long-Short &Delta;:</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm" style={{ color: result.sharpe_ls_pt >= 0 ? "#4ade80" : "#f87171" }}>
                    {result.sharpe_ls_pt > 0 ? "+" : ""}{result.sharpe_ls_pt.toFixed(4)}
                  </span>
                  <SigLabel sig={result.sharpe_ls_sig} />
                </div>
              </div>
            </div>
          </div>

          {/* AUC Bootstrap */}
          {result.auc_pt != null && (
            <div className="card space-y-3">
              <div className="flex items-center justify-between">
                <p className="font-semibold text-[#f5f5f5] text-sm">Bootstrap &Delta;AUC (3000 resamples)</p>
                <SigLabel sig={result.auc_sig} />
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-[#6b6b7a]">&Delta;AUC:</span>
                <span className="font-mono text-base" style={{ color: result.auc_pt > 0 ? "#4ade80" : "#f87171" }}>
                  {result.auc_pt > 0 ? "+" : ""}{result.auc_pt.toFixed(4)}
                </span>
                <SigLabel sig={result.auc_sig} />
              </div>
            </div>
          )}

          {/* Per-period table */}
          {result.per_period && result.per_period.length > 0 && (
            <div className="card">
              <p className="section-title">Per-Year Breakdown</p>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr>
                      <th className="th">Year</th>
                      <th className="th">N</th>
                      <th className="th">Static Acc</th>
                      <th className="th">Adaptive Acc</th>
                      <th className="th">Acc &Delta;</th>
                      <th className="th">Drift Events</th>
                      <th className="th">Sharpe &Delta;</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_period.map((r, i) => {
                      const da = r.acc_adaptive - r.acc_static;
                      const sh = r.sharpe_diff;
                      return (
                        <tr key={i} className="table-row">
                          <td className="td font-mono text-xs">{r.period}</td>
                          <td className="td font-mono text-xs">{r.n_obs}</td>
                          <td className="td font-mono text-xs">{(r.acc_static * 100).toFixed(1)}%</td>
                          <td className="td font-mono text-xs" style={{ color: "#d4a853" }}>{(r.acc_adaptive * 100).toFixed(1)}%</td>
                          <td className="td">
                            <span className="font-mono text-xs" style={{ color: da > 0.005 ? "#4ade80" : da < -0.005 ? "#f87171" : "#6b6b7a" }}>
                              {da > 0 ? "+" : ""}{(da * 100).toFixed(1)}pp
                            </span>
                          </td>
                          <td className="td font-mono text-xs">{r.n_drift_events}</td>
                          <td className="td">
                            <span className="font-mono text-xs" style={{ color: sh != null && sh > 0.05 ? "#4ade80" : sh != null && sh < -0.05 ? "#f87171" : "#6b6b7a" }}>
                              {sh != null ? `${sh > 0 ? "+" : ""}${sh.toFixed(4)}` : "—"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
