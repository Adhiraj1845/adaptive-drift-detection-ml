import React from "react";

interface StatCardProps {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  accent?: "default" | "green" | "red" | "amber" | "gold";
}

const accentColor: Record<string, string> = {
  default: "#27272a",
  green:   "#4ade80",
  red:     "#f87171",
  amber:   "#fb923c",
  gold:    "#d4a853",
};

export function StatCard({ label, value, sub, accent = "default" }: StatCardProps) {
  return (
    <div
      className="card-sm flex flex-col gap-1"
      style={{ borderTopColor: accentColor[accent], borderTopWidth: accent !== "default" ? 1 : 1 }}
    >
      <span className="label">{label}</span>
      <span className="val">{value ?? "—"}</span>
      {sub && <div className="mt-0.5">{sub}</div>}
    </div>
  );
}
