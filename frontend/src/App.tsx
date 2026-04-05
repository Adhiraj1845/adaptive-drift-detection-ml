import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { ActiveStream, Page, RunMeta } from "./types";
import { NavBar } from "./components/NavBar";
import { ActiveRunBanner } from "./components/ActiveRunBanner";
import { Home } from "./pages/Home";
import { NewRun } from "./pages/NewRun";
import { Dashboard } from "./pages/Dashboard";
import { EquityCurves } from "./pages/EquityCurves";
import { DriftAnalysis } from "./pages/DriftAnalysis";
import { Correlations } from "./pages/Correlations";
import { ModelQuality } from "./pages/ModelQuality";
import { StatTests } from "./pages/StatTests";

export default function App() {
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("home");
  const [stream, setStream] = useState<ActiveStream | null>(null);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function refreshRuns(selectTag?: string) {
    api.runs().then((r) => {
      setRuns(r);
      if (selectTag) setSelectedTag(selectTag);
      else if (!selectedTag && r.length > 0) setSelectedTag(r[0].tag);
    });
  }

  useEffect(() => { refreshRuns(); }, []);

  function startStream(runId: string, tag: string) {
    const start = Date.now();
    setStream({ runId, tag, logs: [], progress: 0, elapsed: 0, done: false, error: null });

    if (elapsedRef.current) clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => {
      setStream((prev) => prev ? { ...prev, elapsed: Math.floor((Date.now() - start) / 1000) } : prev);
    }, 1000);

    const es = new EventSource(`/api/run/${runId}/stream`);
    es.onmessage = (e) => {
      const text = e.data as string;
      if (text === "__DONE__") {
        es.close();
        if (elapsedRef.current) clearInterval(elapsedRef.current);
        setStream((prev) => prev ? { ...prev, done: true, progress: 100 } : prev);
        refreshRuns(tag);
        return;
      }
      setStream((prev) => {
        if (!prev) return prev;
        const logs = [...prev.logs, text];
        const match = text.match(/^\[(\d+)\/10\]/);
        const progress = match ? Math.round((parseInt(match[1]) / 10) * 100) : prev.progress;
        return { ...prev, logs, progress };
      });
    };
    es.onerror = () => {
      es.close();
      if (elapsedRef.current) clearInterval(elapsedRef.current);
      setStream((prev) => prev ? { ...prev, error: "Connection lost", done: true } : prev);
    };
  }

  function handleDeleteRun(tag: string) {
    api.deleteRun(tag).then(() => {
      if (selectedTag === tag) {
        setSelectedTag(null);
        setPage("home");
      }
      refreshRuns();
    });
  }

  function goToAnalysis(tag: string, p: Page = "dashboard") {
    setSelectedTag(tag);
    setPage(p);
  }

  const activeRun = runs.find((r) => r.tag === selectedTag) ?? null;
  const analysisTabs: { key: Page; label: string }[] = [
    { key: "dashboard",    label: "Dashboard" },
    { key: "equity",       label: "Equity" },
    { key: "drift",        label: "Drift" },
    { key: "correlations", label: "Correlations" },
    { key: "quality",      label: "Model Quality" },
    { key: "stats",        label: "Significance" },
  ];
  const isAnalysisPage = analysisTabs.some((t) => t.key === page);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-bg">
      <NavBar
        runs={runs}
        selectedTag={selectedTag}
        page={page}
        onPage={(p) => setPage(p)}
        onSelectTag={(tag) => goToAnalysis(tag)}
        activeStream={stream}
      />

      <main className="flex-1 overflow-hidden flex flex-col">
        {/* Tab bar for analysis pages */}
        {isAnalysisPage && activeRun && (
          <div className="shrink-0 flex items-center gap-1 px-6 border-b border-border bg-surface overflow-x-auto">
            <span className="font-mono text-[10px] text-[#3f3f46] pr-3 shrink-0">
              {activeRun.tag}
            </span>
            {analysisTabs.map((t) => (
              <button
                key={t.key}
                onClick={() => setPage(t.key)}
                className={page === t.key ? "tab-active shrink-0" : "tab shrink-0"}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}

        <div className="flex-1 overflow-hidden">
          {page === "home" ? (
            <Home runs={runs} onSelectRun={(tag) => goToAnalysis(tag)} onDeleteRun={handleDeleteRun} onNewRun={() => setPage("new_run")} />
          ) : page === "new_run" ? (
            <NewRun stream={stream} onStartStream={startStream} onDone={(tag) => { goToAnalysis(tag); }} />
          ) : !activeRun ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-4">
              <p className="text-[#6b6b7a] text-sm">No run selected.</p>
              <button onClick={() => setPage("home")} className="btn-primary">Go to Home</button>
            </div>
          ) : page === "dashboard"    ? <Dashboard    run={activeRun} />
          : page === "equity"         ? <EquityCurves run={activeRun} />
          : page === "drift"          ? <DriftAnalysis run={activeRun} />
          : page === "correlations"   ? <Correlations  run={activeRun} />
          : page === "quality"        ? <ModelQuality  run={activeRun} />
          : page === "stats"          ? <StatTests     run={activeRun} />
          : null}
        </div>
      </main>

      {stream && !stream.done && (
        <ActiveRunBanner stream={stream} onView={() => setPage("new_run")} />
      )}
    </div>
  );
}
