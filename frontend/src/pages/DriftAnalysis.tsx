import { useEffect, useMemo, useState } from "react";
import {
  Bar, CartesianGrid, Cell, ComposedChart, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import { TierBadge } from "../components/Badge";
import type { DailyRow, DriftEvent, RunMeta } from "../types";

const C = {
  static:   "#52525b",
  adaptive: "#d4a853",
  feat:     "#60a5fa",
  pred:     "#d4a853",
  perf:     "#f87171",
  event:    "#fb923c",
  grid:     "#27272a",
};

const TIER_COLORS: Record<string, string> = {
  moderate: "#d4a853",
  high:     "#fb923c",
  severe:   "#f87171",
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
        <p key={e.name} style={{ color: e.color ?? "#a1a1aa", fontSize: 11, fontFamily: "DM Mono, monospace" }}>
          {e.name}: {typeof e.value === "number" ? e.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

function rolling(data: (number | null | undefined)[], w: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < w - 1) return null;
    const s = data.slice(i - w + 1, i + 1).filter((v) => v != null) as number[];
    return s.length ? s.reduce((a, b) => a + b, 0) / s.length : null;
  });
}

interface Props { run: RunMeta; }

export function DriftAnalysis({ run }: Props) {
  const [daily,   setDaily]   = useState<DailyRow[]>([]);
  const [events,  setEvents]  = useState<DriftEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [window_, setWindow]  = useState(60);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.daily(run.tag), api.driftEvents(run.tag)]).then(([d, ev]) => {
      setDaily(d); setEvents(ev); setLoading(false);
    });
  }, [run.tag]);

  const rollLoss  = useMemo(() => rolling(daily.map((d) => d.logloss_static),   window_), [daily, window_]);
  const rollLossA = useMemo(() => rolling(daily.map((d) => d.logloss_adaptive), window_), [daily, window_]);
  const rollAcc   = useMemo(() => rolling(
    daily.map((d) => d.y_true_next != null && d.y_pred_static != null
      ? (d.y_pred_static === d.y_true_next ? 1 : 0) : null), window_
  ), [daily, window_]);
  const rollAccA  = useMemo(() => rolling(
    daily.map((d) => d.y_true_next != null && d.y_pred_adaptive != null
      ? (d.y_pred_adaptive === d.y_true_next ? 1 : 0) : null), window_
  ), [daily, window_]);

  const driftDates = useMemo(() => new Set(events.map((ev) => ev.date?.slice(0, 10))), [events]);

  const chartData = daily.map((r, i) => ({
    date:      r.date?.slice(0, 10),
    drift_idx: r.drift_index,
    feat:      r.feature_score,
    pred:      r.prediction_score,
    perf:      r.performance_score,
    loss_s:    rollLoss[i],
    loss_a:    rollLossA[i],
    acc_s:     rollAcc[i],
    acc_a:     rollAccA[i],
    is_event:  r.drift_event === 1 ? (r.drift_index ?? 0) : null,
  }));

  const tierCounts = useMemo(() => {
    const c: Record<string, number> = { moderate: 0, high: 0, severe: 0 };
    events.forEach((ev) => {
      const t = (ev.action ?? (ev["tier"] as string | undefined) ?? "").toLowerCase();
      if (t in c) c[t]++;
    });
    return c;
  }, [events]);

  if (loading) return (
    <div className="flex-1 flex items-center justify-center font-mono text-[12px] text-[#52525b]">Loading…</div>
  );

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-[#fafafa] tracking-[-0.3px]">Drift Analysis</h1>
          <p className="font-mono text-[11px] text-[#52525b] mt-1">
            {events.length} events detected over {daily.length} days
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-4">
            {Object.entries(tierCounts).map(([t, n]) => (
              <div key={t} className="text-center">
                <div className="font-mono text-[9px] text-[#52525b] uppercase tracking-[1px]">{t}</div>
                <div className="font-mono text-[15px] font-semibold" style={{ color: TIER_COLORS[t] }}>{n}</div>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-1.5 ml-2">
            {[30, 60, 90].map((w) => (
              <button key={w} onClick={() => setWindow(w)} className={window_ === w ? "btn-active text-xs py-1 px-2" : "btn-ghost text-xs py-1 px-2"}>
                {w}d
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Composite drift index */}
      <div className="card">
        <p className="section-title">Composite Drift Index</p>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={chartData}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={45} />
            <Tooltip content={<Tip />} />
            <Bar  dataKey="is_event"  name="Event"       fill={C.event} opacity={0.5} maxBarSize={3} />
            <Line dataKey="drift_idx" name="Drift Index" stroke={C.adaptive} dot={false} strokeWidth={1.5} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Component scores */}
      <div className="card">
        <p className="section-title">Detector Component Scores</p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={45} />
            <Tooltip content={<Tip />} />
            <Line dataKey="feat" name="Feature"     stroke={C.feat} dot={false} strokeWidth={1.2} />
            <Line dataKey="pred" name="Prediction"  stroke={C.pred} dot={false} strokeWidth={1.2} />
            <Line dataKey="perf" name="Performance" stroke={C.perf} dot={false} strokeWidth={1.2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Rolling log-loss and accuracy */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="card">
          <p className="section-title">{window_}-Day Rolling Log-Loss</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
              <YAxis tick={AXIS} tickLine={false} axisLine={false} width={50} />
              <Tooltip content={<Tip />} />
              {[...driftDates].slice(0, 40).map((d) => (
                <ReferenceLine key={d} x={d} stroke={C.event} strokeOpacity={0.2} strokeWidth={1} />
              ))}
              <Line dataKey="loss_s" name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
              <Line dataKey="loss_a" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <p className="section-title">{window_}-Day Rolling Accuracy</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
              <YAxis tick={AXIS} tickLine={false} axisLine={false} domain={[0.3, 0.7]} width={45} />
              <Tooltip content={<Tip />} />
              <ReferenceLine y={0.5} stroke="#3f3f46" strokeDasharray="4 4" strokeOpacity={0.8} />
              {[...driftDates].slice(0, 40).map((d) => (
                <ReferenceLine key={d} x={d} stroke={C.event} strokeOpacity={0.15} strokeWidth={1} />
              ))}
              <Line dataKey="acc_s" name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
              <Line dataKey="acc_a" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Event log */}
      {events.length > 0 && (
        <div className="card">
          <p className="section-title">Drift Event Log ({events.length})</p>
          <div className="overflow-y-auto max-h-72">
            <table className="w-full">
              <thead className="sticky top-0 bg-surface">
                <tr>
                  <th className="th">#</th>
                  <th className="th">Date</th>
                  <th className="th">Tier</th>
                  <th className="th">Index</th>
                  <th className="th">Feature</th>
                  <th className="th">Prediction</th>
                  <th className="th">Performance</th>
                </tr>
              </thead>
              <tbody>
                {events.map((ev, i) => (
                  <tr key={i} className="table-row">
                    <td className="td font-mono text-[#52525b]">{i + 1}</td>
                    <td className="td font-mono">{ev.date?.slice(0, 10)}</td>
                    <td className="td"><TierBadge tier={ev.action ?? (ev["tier"] as string | undefined)} /></td>
                    <td className="td font-mono">{ev.drift_index?.toFixed(4) ?? "—"}</td>
                    <td className="td font-mono">{ev.feature_score?.toFixed(4) ?? "—"}</td>
                    <td className="td font-mono">{ev.prediction_score?.toFixed(4) ?? "—"}</td>
                    <td className="td font-mono">{ev.performance_score?.toFixed(4) ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
