import type { ActiveStream } from "../types";

interface Props {
  stream: ActiveStream;
  onView: () => void;
}

function fmt(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function ActiveRunBanner({ stream, onView }: Props) {
  const lastLog = [...stream.logs].reverse().find((l) => l.trim());

  return (
    <div className="shrink-0 h-12 flex items-center px-6 gap-4 bg-[#161618] border-t border-[#2a2a2d]">
      <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] animate-pulse shrink-0" />
      <span className="font-mono text-[11px] text-[#f5f5f5] shrink-0 hidden sm:block">
        {stream.tag.split("_").slice(0,2).join("_")}
      </span>

      {/* Progress bar */}
      <div className="flex-1 h-1 bg-[#2a2a2d] rounded-full overflow-hidden max-w-xs">
        <div
          className="h-full bg-[#d4a853] rounded-full transition-all duration-500"
          style={{ width: `${stream.progress}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-[#6b6b7a] shrink-0">{stream.progress}%</span>

      {/* Elapsed */}
      <span className="font-mono text-[10px] text-[#6b6b7a] shrink-0">&#9201; {fmt(stream.elapsed)}</span>

      {/* Last log line */}
      {lastLog && (
        <span className="font-mono text-[10px] text-[#6b6b7a] truncate flex-1 hidden md:block">
          {lastLog.replace(/^\[\d+\/10\]\s*/, "")}
        </span>
      )}

      <button onClick={onView} className="btn-ghost text-xs shrink-0">
        View &rarr;
      </button>
    </div>
  );
}
