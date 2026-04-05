interface BadgeProps {
  sig: boolean | null | undefined;
  pval?: number | null;
  label?: string;
}

export function SigBadge({ sig, pval, label }: BadgeProps) {
  if (sig === null || sig === undefined) return <span className="pill-slate">n/a</span>;
  if (sig) {
    return (
      <span className="pill-green">
        ✓ sig{pval != null ? ` p=${pval.toFixed(3)}` : ""}
      </span>
    );
  }
  return (
    <span className="pill-slate">
      not sig{pval != null ? ` p=${pval.toFixed(3)}` : ""}
      {label ? ` — ${label}` : ""}
    </span>
  );
}

export function TierBadge({ tier }: { tier?: string }) {
  const t = tier?.toLowerCase() ?? "none";
  if (t === "severe")   return <span className="pill-red">severe</span>;
  if (t === "high")     return <span className="pill-amber">high</span>;
  if (t === "moderate") return <span className="pill-gold">moderate</span>;
  return <span className="pill-slate">none</span>;
}

export function AccDelta({ s, a }: { s?: number; a?: number }) {
  if (s == null || a == null) return null;
  const d = a - s;
  const color = d > 0.005 ? "#4ade80" : d < -0.005 ? "#f87171" : "#71717a";
  const arrow = d > 0.005 ? "▲" : d < -0.005 ? "▼" : "—";
  return (
    <span className="font-mono text-[11px]" style={{ color }}>
      {arrow} {Math.abs(d * 100).toFixed(1)}pp
    </span>
  );
}
