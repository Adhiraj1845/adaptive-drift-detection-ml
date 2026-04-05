import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import type { EquityRow, RunMeta } from "../types";

type Mode = "longonly" | "longshort";

const C = {
  market:   "#60a5fa",
  static:   "#52525b",
  adaptive: "#d4a853",
  grid:     "#27272a",
};

const AXIS = { fill: "#52525b", fontSize: 10, fontFamily: "DM Mono, monospace" };

const TIP_STYLE = {
  background: "#18181b", border: "1px solid #27272a", borderRadius: 8, padding: "10px 14px",
};

function Tip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={TIP_STYLE}>
      <p style={{ color: "#52525b", fontSize: 10, fontFamily: "DM Mono, monospace", marginBottom: 4 }}>{label}</p>
      {payload.map((e: any) => (
        <p key={e.name} style={{ color: e.color, fontSize: 11, fontFamily: "DM Mono, monospace" }}>
          {e.name}: {typeof e.value === "number" ? e.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

interface Props { run: RunMeta; }

export function EquityCurves({ run }: Props) {
  const [data, setData]     = useState<EquityRow[]>([]);
  const [mode, setMode]     = useState<Mode>("longonly");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.equity(run.tag).then((d) => { setData(d); setLoading(false); });
  }, [run.tag]);

  const sKey = mode === "longonly" ? "equity_longonly_static"   : "equity_longshort_static";
  const aKey = mode === "longonly" ? "equity_longonly_adaptive" : "equity_longshort_adaptive";

  const chartData = data.map((r) => ({
    date:     r.date?.slice(0, 10),
    market:   r.equity_market,
    static:   r[sKey] as number | undefined,
    adaptive: r[aKey] as number | undefined,
  }));

  const last  = chartData.at(-1);
  const first = chartData.find((r) => r.market != null);

  function ret(end?: number, start?: number) {
    if (end == null || start == null || start === 0) return null;
    return ((end / start) - 1) * 100;
  }

  const mktRet = ret(last?.market,   first?.market);
  const sRet   = ret(last?.static,   first?.static);
  const aRet   = ret(last?.adaptive, first?.adaptive);

  if (loading) return (
    <div className="flex-1 flex items-center justify-center font-mono text-[12px] text-[#52525b]">
      Loading…
    </div>
  );

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-[#fafafa] tracking-[-0.3px]">Equity Curves</h1>
          <p className="font-mono text-[11px] text-[#52525b] mt-1">Cumulative portfolio value starting at 1.0</p>
        </div>
        <div className="flex gap-2">
          {(["longonly", "longshort"] as Mode[]).map((m) => (
            <button key={m} onClick={() => setMode(m)} className={mode === m ? "btn-active" : "btn-ghost"}>
              {m === "longonly" ? "Long-Only" : "Long-Short"}
            </button>
          ))}
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Market Return",   val: mktRet, color: C.market },
          { label: "Static Return",   val: sRet,   color: sRet  != null ? (sRet  >= 0 ? "#4ade80" : "#f87171") : C.static },
          { label: "Adaptive Return", val: aRet,   color: aRet  != null ? (aRet  >= 0 ? "#4ade80" : "#f87171") : C.adaptive },
        ].map(({ label, val, color }) => (
          <div key={label} className="card-sm flex flex-col gap-1">
            <span className="label">{label}</span>
            <span className="val" style={{ color }}>
              {val != null ? `${val > 0 ? "+" : ""}${val.toFixed(1)}%` : "—"}
            </span>
          </div>
        ))}
      </div>

      {/* Main chart */}
      <div className="card">
        <p className="section-title">Portfolio Value — {mode === "longonly" ? "Long Only" : "Long-Short"}</p>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={55} />
            <Tooltip content={<Tip />} />
            <Line dataKey="market"   name="Market"   stroke={C.market}   dot={false} strokeWidth={1.2} strokeDasharray="5 3" />
            <Line dataKey="static"   name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
            <Line dataKey="adaptive" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Relative performance */}
      {chartData.some((r) => r.static != null && r.adaptive != null) && (
        <div className="card">
          <p className="section-title">Adaptive / Static (relative performance)</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData.map((r) => ({
              date: r.date,
              rel: r.static && r.adaptive ? r.adaptive / r.static : null,
            }))}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
              <YAxis tick={AXIS} tickLine={false} axisLine={false} width={55} />
              <Tooltip content={<Tip />} />
              <Line dataKey="rel" name="Adaptive/Static" stroke={C.adaptive} dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
