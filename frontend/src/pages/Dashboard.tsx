import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import type { DailyRow, EquityRow, EvalResult, RunMeta } from "../types";

const C = { static: "#6b6b7a", adaptive: "#d4a853", market: "#60a5fa", grid: "#2a2a2d" };
const AXIS = { fill: "#6b6b7a", fontSize: 10, fontFamily: "DM Mono, monospace" };
const TIP_STYLE = { background: "#1e1e21", border: "1px solid #2a2a2d", borderRadius: 8, padding: "10px 14px" };

function Tip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={TIP_STYLE}>
      <p style={{ color: "#6b6b7a", fontSize: 10, fontFamily: "DM Mono, monospace", marginBottom: 4 }}>{label}</p>
      {payload.map((e: any) => (
        <p key={e.name} style={{ color: e.color, fontSize: 11, fontFamily: "DM Mono, monospace" }}>
          {e.name}: {typeof e.value === "number" ? e.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

interface Props { run: RunMeta; }

export function Dashboard({ run }: Props) {
  const [daily, setDaily] = useState<DailyRow[]>([]);
  const [equity, setEquity] = useState<EquityRow[]>([]);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [evalLoading, setEvalLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    setEvalResult(null);
    Promise.all([api.daily(run.tag), api.equity(run.tag)]).then(([d, e]) => {
      setDaily(d);
      setEquity(e);
      setLoading(false);
    });
    // Auto-load evaluation
    setEvalLoading(true);
    api.evaluate(run.tag).then((r) => { setEvalResult(r); setEvalLoading(false); }).catch(() => setEvalLoading(false));
  }, [run.tag]);

  const equityData = equity.map((r) => ({
    date: r.date?.slice(0, 10),
    market: r.equity_market,
    static: r.equity_longonly_static,
    adaptive: r.equity_longonly_adaptive,
  }));

  // Rolling 30-day log-loss
  function rolling30(arr: (number | undefined | null)[]): (number | null)[] {
    return arr.map((_, i) => {
      if (i < 29) return null;
      const s = arr.slice(i - 29, i + 1).filter((v) => v != null) as number[];
      return s.length ? s.reduce((a, b) => a + b, 0) / s.length : null;
    });
  }
  const lossS = rolling30(daily.map((d) => d.logloss_static));
  const lossA = rolling30(daily.map((d) => d.logloss_adaptive));
  const lossData = daily.map((r, i) => ({ date: r.date?.slice(0, 10), s: lossS[i], a: lossA[i] }));

  const accDelta = run.acc_adaptive != null && run.acc_static != null
    ? (run.acc_adaptive - run.acc_static) * 100
    : null;

  const lastEquity = equityData.at(-1);
  const firstEquity = equityData.find((r) => r.market != null);
  function ret(e?: number, s?: number) {
    if (e == null || s == null || s === 0) return null;
    return ((e / s) - 1) * 100;
  }
  const mktRet = ret(lastEquity?.market, firstEquity?.market);
  const aRet = ret(lastEquity?.adaptive, firstEquity?.adaptive);

  if (loading) return <div className="flex-1 flex items-center justify-center font-mono text-sm text-[#6b6b7a]">Loading…</div>;

  // Recent drift events
  const driftDays = daily.filter((d) => d.action && d.action !== "none").slice(-10).reverse();

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Key stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="card-sm flex flex-col gap-1">
          <span className="label">Observations</span>
          <span className="font-mono text-xl text-[#f5f5f5]">{run.n_obs?.toLocaleString() ?? "—"}</span>
        </div>
        <div className="card-sm flex flex-col gap-1">
          <span className="label">Drift Events</span>
          <span className="font-mono text-xl" style={{ color: run.n_drift_events ? "#fb923c" : "#6b6b7a" }}>
            {run.n_drift_events ?? "—"}
          </span>
        </div>
        <div className="card-sm flex flex-col gap-1">
          <span className="label">Adaptive Acc</span>
          <span className="font-mono text-xl" style={{ color: "#d4a853" }}>
            {run.acc_adaptive != null ? `${(run.acc_adaptive * 100).toFixed(1)}%` : "—"}
          </span>
          {accDelta != null && (
            <span className="font-mono text-[10px]" style={{ color: accDelta > 0 ? "#4ade80" : accDelta < 0 ? "#f87171" : "#6b6b7a" }}>
              {accDelta > 0 ? "+" : ""}{accDelta.toFixed(1)}pp vs static
            </span>
          )}
        </div>
        <div className="card-sm flex flex-col gap-1">
          <span className="label">Adaptive Return</span>
          <span className="font-mono text-xl" style={{ color: aRet != null ? (aRet >= 0 ? "#4ade80" : "#f87171") : "#6b6b7a" }}>
            {aRet != null ? `${aRet > 0 ? "+" : ""}${aRet.toFixed(1)}%` : "—"}
          </span>
          {mktRet != null && (
            <span className="font-mono text-[10px] text-[#6b6b7a]">mkt: {mktRet > 0 ? "+" : ""}{mktRet.toFixed(1)}%</span>
          )}
        </div>
      </div>

      {/* Eval summary */}
      {evalLoading && (
        <div className="card font-mono text-xs text-[#6b6b7a] text-center py-4">
          Computing statistical tests…
        </div>
      )}
      {evalResult && !evalLoading && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            {
              label: "McNemar p",
              val: evalResult.mcnemar_p.toFixed(4),
              color: evalResult.mcnemar_sig_bonf ? "#4ade80" : "#f87171",
              note: evalResult.mcnemar_sig_bonf ? "significant" : "not significant",
            },
            {
              label: "OLS β",
              val: `${evalResult.ols_beta > 0 ? "+" : ""}${evalResult.ols_beta.toFixed(4)}`,
              color: "#6b6b7a",
              note: `p = ${evalResult.ols_p.toFixed(4)}`,
            },
            {
              label: "Sharpe Δ (LO)",
              val: `${evalResult.sharpe_long_pt > 0 ? "+" : ""}${evalResult.sharpe_long_pt.toFixed(4)}`,
              color: evalResult.sharpe_long_pt >= 0 ? "#4ade80" : "#f87171",
              note: evalResult.sharpe_long_sig ? "significant" : "not significant",
            },
            {
              label: "ΔAUC",
              val: evalResult.auc_pt != null ? `${evalResult.auc_pt > 0 ? "+" : ""}${evalResult.auc_pt.toFixed(4)}` : "—",
              color: evalResult.auc_pt != null ? (evalResult.auc_pt >= 0 ? "#4ade80" : "#f87171") : "#6b6b7a",
              note: evalResult.auc_sig ? "significant" : "not significant",
            },
          ].map(({ label, val, color, note }) => (
            <div key={label} className="card-sm flex flex-col gap-1">
              <span className="label">{label}</span>
              <span className="font-mono text-xl" style={{ color, fontVariantNumeric: "tabular-nums" }}>{val}</span>
              <span className="font-mono text-[10px] text-[#6b6b7a]">{note}</span>
            </div>
          ))}
        </div>
      )}

      {/* Equity chart */}
      <div className="card">
        <p className="section-title">Equity Curves (Long-Only)</p>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={equityData} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={50} />
            <Tooltip content={<Tip />} />
            <Line dataKey="market"   name="Market"   stroke={C.market}   dot={false} strokeWidth={1.2} strokeDasharray="4 3" />
            <Line dataKey="static"   name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
            <Line dataKey="adaptive" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Rolling log-loss */}
      <div className="card">
        <p className="section-title">30-Day Rolling Log-Loss</p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={lossData}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={45} />
            <Tooltip content={<Tip />} />
            <Line dataKey="s" name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
            <Line dataKey="a" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent drift events */}
      {driftDays.length > 0 && (
        <div className="card">
          <p className="section-title">Recent Drift Events</p>
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Date</th>
                <th className="th">Action</th>
                <th className="th">Drift Index</th>
                <th className="th">Feature Score</th>
              </tr>
            </thead>
            <tbody>
              {driftDays.map((r, i) => {
                const tierColor = r.action === "severe" ? "#f87171" : r.action === "high" ? "#fb923c" : "#d4a853";
                return (
                  <tr key={i} className="table-row">
                    <td className="td font-mono text-xs">{r.date?.slice(0,10)}</td>
                    <td className="td">
                      <span className="pill font-mono text-[10px]" style={{ background: tierColor + "20", color: tierColor }}>
                        {r.action}
                      </span>
                    </td>
                    <td className="td font-mono text-xs">{r.drift_index?.toFixed(4) ?? "—"}</td>
                    <td className="td font-mono text-xs">{r.feature_score?.toFixed(4) ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
