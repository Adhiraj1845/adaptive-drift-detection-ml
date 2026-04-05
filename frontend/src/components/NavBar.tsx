import { ChevronDown } from "lucide-react";
import { useState } from "react";
import type { ActiveStream, Page, RunMeta } from "../types";

interface Props {
  runs: RunMeta[];
  selectedTag: string | null;
  page: Page;
  onPage: (p: Page) => void;
  onSelectTag: (tag: string) => void;
  activeStream: ActiveStream | null;
}

export function NavBar({ runs, selectedTag, page, onPage, onSelectTag, activeStream }: Props) {
  const [runsOpen, setRunsOpen] = useState(false);

  function fmt(tag: string) {
    // Show just the ticker part before the first underscore for brevity
    const parts = tag.split("_");
    return parts.slice(0, 2).join("_");
  }

  return (
    <nav className="shrink-0 h-14 flex items-center px-6 gap-4 border-b border-border bg-surface z-50">
      {/* Logo */}
      <button
        onClick={() => onPage("home")}
        className="flex items-center gap-2 mr-2 shrink-0"
      >
        <span className="text-gold font-mono font-bold text-sm tracking-tight">⚡ DRIFT</span>
      </button>

      {/* Nav links */}
      <button
        onClick={() => onPage("home")}
        className={page === "home" ? "nav-link-active" : "nav-link"}
      >
        Home
      </button>
      <button
        onClick={() => onPage("new_run")}
        className={page === "new_run" ? "nav-link-active" : "nav-link"}
      >
        New Run
      </button>

      {/* Runs dropdown */}
      {runs.length > 0 && (
        <div className="relative">
          <button
            onClick={() => setRunsOpen((o) => !o)}
            className="nav-link flex items-center gap-1"
          >
            {selectedTag ? (
              <span className="font-mono text-xs text-[#f5f5f5] max-w-[200px] truncate">{fmt(selectedTag)}</span>
            ) : (
              "Select Run"
            )}
            <ChevronDown size={13} className="opacity-50" />
          </button>

          {runsOpen && (
            <div className="absolute top-full mt-1 left-0 w-72 bg-[#1e1e21] border border-[#2a2a2d] rounded-xl shadow-2xl z-50 overflow-hidden">
              <div className="p-2 max-h-80 overflow-y-auto">
                {runs.map((r) => (
                  <button
                    key={r.tag}
                    onClick={() => { onSelectTag(r.tag); setRunsOpen(false); }}
                    className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex flex-col gap-0.5 ${
                      r.tag === selectedTag ? "bg-[#2a2a2d]" : "hover:bg-[#2a2a2d]/50"
                    }`}
                  >
                    <span className="font-mono text-xs text-[#f5f5f5] truncate">{r.tag}</span>
                    <span className="text-[10px] text-[#6b6b7a] font-mono">
                      {r.date_min?.slice(0,10)} &rarr; {r.date_max?.slice(0,10)}
                      {r.n_obs != null ? `  ·  ${r.n_obs.toLocaleString()} obs` : ""}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Active stream indicator */}
      {activeStream && !activeStream.done && (
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] animate-pulse" />
          <span className="font-mono text-[11px] text-[#6b6b7a] hidden sm:block">
            {activeStream.tag.split("_").slice(0,2).join("_")} · {activeStream.progress}%
          </span>
        </div>
      )}
    </nav>
  );
}
