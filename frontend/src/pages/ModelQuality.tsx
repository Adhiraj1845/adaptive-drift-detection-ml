import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";
import type { DailyRow, RocPrData, RunMeta } from "../types";

const C = {
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
      <p style={{ color: "#52525b", fontSize: 10, fontFamily: "DM Mono, monospace", marginBottom: 4 }}>
        {label?.toFixed?.(3) ?? label}
      </p>
      {payload.map((e: any) => (
        <p key={e.name} style={{ color: e.color, fontSize: 11, fontFamily: "DM Mono, monospace" }}>
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

export function ModelQuality({ run }: Props) {
  const [curves,  setCurves]  = useState<RocPrData | null>(null);
  const [daily,   setDaily]   = useState<DailyRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.rocPr(run.tag).catch(() => null), api.daily(run.tag)])
      .then(([c, d]) => { setCurves(c); setDaily(d); setLoading(false); });
  }, [run.tag]);

  const rollLoss30  = rolling(daily.map((d) => d.logloss_static),   30);
  const rollLossA30 = rolling(daily.map((d) => d.logloss_adaptive), 30);
  const rollLossData = daily.map((r, i) => ({
    date: r.date?.slice(0, 10),
    s:    rollLoss30[i],
    a:    rollLossA30[i],
  }));

  const calibData = (() => {
    const bins = 10;
    const bdata = Array.from({ length: bins }, (_, i) => ({
      mid: (i + 0.5) / bins,
      sum_pred_s: 0, cnt_s: 0, sum_act_s: 0,
      sum_pred_a: 0, cnt_a: 0, sum_act_a: 0,
    }));
    daily.forEach((r) => {
      const y = r.y_true_next;
      if (y == null) return;
      if (r.p1_static != null) {
        const bi = Math.min(bins - 1, Math.floor(r.p1_static * bins));
        bdata[bi].sum_pred_s += r.p1_static;
        bdata[bi].sum_act_s  += y;
        bdata[bi].cnt_s++;
      }
      if (r.p1_adaptive != null) {
        const bi = Math.min(bins - 1, Math.floor(r.p1_adaptive * bins));
        bdata[bi].sum_pred_a += r.p1_adaptive;
        bdata[bi].sum_act_a  += y;
        bdata[bi].cnt_a++;
      }
    });
    return bdata.map((b) => ({
      x:      b.mid,
      frac_s: b.cnt_s > 0 ? b.sum_act_s / b.cnt_s : null,
      frac_a: b.cnt_a > 0 ? b.sum_act_a / b.cnt_a : null,
    }));
  })();

  if (loading) return (
    <div className="flex-1 flex items-center justify-center font-mono text-[12px] text-[#52525b]">Loading…</div>
  );

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#fafafa] tracking-[-0.3px]">Model Quality</h1>
        <p className="font-mono text-[11px] text-[#52525b] mt-1">ROC, precision-recall, calibration, rolling loss</p>
      </div>

      {/* AUC summary */}
      {curves && (
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "AUC — Static",   val: curves.auc_static,   color: C.static },
            { label: "AUC — Adaptive", val: curves.auc_adaptive, color: C.adaptive },
          ].map(({ label, val, color }) => (
            <div key={label} className="card-sm flex flex-col gap-1">
              <span className="label">{label}</span>
              <span className="val" style={{ color }}>{val.toFixed(4)}</span>
              <span className="font-mono text-[10px] text-[#52525b]">area under ROC curve</span>
            </div>
          ))}
        </div>
      )}

      {curves ? (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div className="card">
            <p className="section-title">ROC Curve</p>
            <p className="font-mono text-[10px] text-[#52525b] mb-3">TPR vs FPR across all thresholds</p>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={AXIS} tickLine={false}
                  label={{ value: "FPR", position: "insideBottom", offset: -10, fill: "#52525b", fontSize: 10 }} />
                <YAxis type="number" domain={[0, 1]} tick={AXIS} tickLine={false} axisLine={false} width={40}
                  label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#52525b", fontSize: 10 }} />
                <Tooltip content={<Tip />} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#3f3f46" strokeDasharray="4 4" />
                <Line data={curves.roc_static}   dataKey="tpr" name={`Static (AUC=${curves.auc_static.toFixed(3)})`}   stroke={C.static}   dot={false} strokeWidth={1.5} />
                <Line data={curves.roc_adaptive} dataKey="tpr" name={`Adaptive (AUC=${curves.auc_adaptive.toFixed(3)})`} stroke={C.adaptive} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <p className="section-title">Precision-Recall Curve</p>
            <p className="font-mono text-[10px] text-[#52525b] mb-3">Precision vs recall across all thresholds</p>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis dataKey="rec"  type="number" domain={[0, 1]} tick={AXIS} tickLine={false}
                  label={{ value: "Recall", position: "insideBottom", offset: -10, fill: "#52525b", fontSize: 10 }} />
                <YAxis type="number" domain={[0, 1]} tick={AXIS} tickLine={false} axisLine={false} width={40}
                  label={{ value: "Precision", angle: -90, position: "insideLeft", fill: "#52525b", fontSize: 10 }} />
                <Tooltip content={<Tip />} />
                <Line data={curves.pr_static}   dataKey="prec" name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
                <Line data={curves.pr_adaptive} dataKey="prec" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="card font-mono text-[11px] text-[#52525b] py-8 text-center">
          ROC/PR unavailable — p1_static or p1_adaptive columns not found.
        </div>
      )}

      {daily.some((r) => r.p1_static != null) && (
        <div className="card">
          <p className="section-title">Reliability Diagram (Calibration)</p>
          <p className="font-mono text-[10px] text-[#52525b] mb-3">
            Fraction of positives vs mean predicted probability — perfect calibration is the dashed diagonal.
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={calibData} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid stroke={C.grid} />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tick={AXIS} tickLine={false}
                label={{ value: "Mean Predicted Probability", position: "insideBottom", offset: -10, fill: "#52525b", fontSize: 10 }} />
              <YAxis type="number" domain={[0, 1]} tick={AXIS} tickLine={false} axisLine={false} width={40} />
              <Tooltip content={<Tip />} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#3f3f46" strokeDasharray="4 4" />
              <Line dataKey="frac_s" name="Static"   stroke={C.static}   dot={{ fill: C.static,   r: 3 }} strokeWidth={1.5} connectNulls={false} />
              <Line dataKey="frac_a" name="Adaptive" stroke={C.adaptive} dot={{ fill: C.adaptive, r: 3 }} strokeWidth={2}   connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="card">
        <p className="section-title">30-Day Rolling Log-Loss</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rollLossData}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={50} />
            <Tooltip content={<Tip />} />
            <Line dataKey="s" name="Static"   stroke={C.static}   dot={false} strokeWidth={1.5} />
            <Line dataKey="a" name="Adaptive" stroke={C.adaptive} dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
