import { useEffect, useState } from "react";
import { api } from "../api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import type { CorrelationData, RunMeta } from "../types";

interface Props { run: RunMeta; }

const TIP_STYLE = { background: "#1e1e21", border: "1px solid #2a2a2d", borderRadius: 8, padding: "10px 14px" };
const AXIS = { fill: "#6b6b7a", fontSize: 10, fontFamily: "DM Mono, monospace" };

function Tip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={TIP_STYLE}>
      <p style={{ color: "#f5f5f5", fontSize: 11, fontFamily: "DM Mono, monospace" }}>{d.pair}</p>
      <p style={{ color: d.r >= 0 ? "#4ade80" : "#f87171", fontSize: 11, fontFamily: "DM Mono, monospace" }}>
        r = {d.r > 0 ? "+" : ""}{d.r.toFixed(4)}
      </p>
    </div>
  );
}

export function Correlations({ run }: Props) {
  const [data, setData] = useState<CorrelationData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.correlations(run.tag).then((d) => { setData(d); setLoading(false); });
  }, [run.tag]);

  if (loading) return (
    <div className="flex-1 flex items-center justify-center font-mono text-sm text-[#6b6b7a]">Loading…</div>
  );
  if (!data || !data.columns.length) return (
    <div className="flex-1 flex items-center justify-center font-mono text-sm text-[#6b6b7a]">No numeric columns found.</div>
  );

  // Build all unique pairs sorted by |r|
  const pairs: { pair: string; a: string; b: string; r: number }[] = [];
  data.columns.forEach((a, i) => {
    data.columns.slice(i + 1).forEach((b, j) => {
      const r = data.matrix[i][i + 1 + j];
      if (r != null && !isNaN(r as number) && Math.abs(r as number) > 0.01) {
        pairs.push({ pair: `${a} / ${b}`, a, b, r: r as number });
      }
    });
  });
  const sorted = pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, 20);
  const top10 = sorted.slice(0, 10);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#f5f5f5] tracking-tight">Correlations</h1>
        <p className="font-mono text-[11px] text-[#6b6b7a] mt-1">Pearson r — top feature pairs by absolute correlation</p>
      </div>

      {/* Bar chart */}
      <div className="card">
        <p className="section-title">Top {top10.length} Pairs by |r|</p>
        <ResponsiveContainer width="100%" height={Math.max(200, top10.length * 32)}>
          <BarChart data={top10} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 120 }}>
            <CartesianGrid stroke="#2a2a2d" horizontal={false} />
            <XAxis type="number" domain={[-1, 1]} tick={AXIS} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="pair" tick={{ fill: "#6b6b7a", fontSize: 10, fontFamily: "DM Mono, monospace" }} tickLine={false} axisLine={false} width={120} />
            <Tooltip content={<Tip />} />
            <ReferenceLine x={0} stroke="#3f3f46" />
            <Bar dataKey="r" radius={[0, 3, 3, 0]}>
              {top10.map((entry, i) => (
                <Cell key={i} fill={entry.r >= 0 ? "#4ade80" : "#f87171"} opacity={0.7 + 0.3 * (Math.abs(entry.r))} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Full table */}
      <div className="card">
        <p className="section-title">All Pairs (sorted by |r|)</p>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Feature A</th>
                <th className="th">Feature B</th>
                <th className="th">Pearson r</th>
                <th className="th">Strength</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(({ pair: _pair, a, b, r }, i) => {
                const abs = Math.abs(r);
                const strength = abs > 0.7 ? "strong" : abs > 0.4 ? "moderate" : "weak";
                const cls = abs > 0.7 ? "pill-green" : abs > 0.4 ? "pill-gold" : "pill-slate";
                return (
                  <tr key={i} className="table-row">
                    <td className="td font-mono text-xs">{a}</td>
                    <td className="td font-mono text-xs">{b}</td>
                    <td className="td">
                      <span className="font-mono text-sm" style={{ color: r > 0 ? "#4ade80" : "#f87171" }}>
                        {r > 0 ? "+" : ""}{r.toFixed(4)}
                      </span>
                    </td>
                    <td className="td"><span className={cls}>{strength}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
