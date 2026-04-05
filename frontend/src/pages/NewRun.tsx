import { useState } from "react";
import { api } from "../api";
import type { ActiveStream } from "../types";

interface Props {
  stream: ActiveStream | null;
  onStartStream: (runId: string, tag: string) => void;
  onDone: (tag: string) => void;
}

const MODELS = [
  { value: "gradient_boosting", label: "Gradient Boosting (fast, recommended)" },
  { value: "random_forest", label: "Random Forest (slow — ~10 min)" },
  { value: "logistic_regression", label: "Logistic Regression (fastest)" },
];

const SOURCES = [
  { value: "yahoo", label: "Yahoo Finance (ticker, e.g. ^GSPC)" },
  { value: "fred", label: "FRED (series, e.g. DGS10)" },
];

function logColor(line: string) {
  if (line.includes("Error") || line.includes("error") || line.includes("Traceback")) return "#f87171";
  if (line.includes("Done") || line.includes("done") || line.includes("__DONE__") || line.includes("Saved")) return "#4ade80";
  if (line.includes("Warning") || line.includes("warn")) return "#fb923c";
  if (line.includes("[") && line.includes("/10]")) return "#d4a853";
  return "#a1a1aa";
}

function fmt(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function NewRun({ stream, onStartStream, onDone }: Props) {
  const [step, setStep] = useState(0);
  const [source, setSource] = useState("yahoo");
  const [ticker, setTicker] = useState("^GSPC");
  const [dataStart, setDataStart] = useState("2015-01-01");
  const [dataEnd, setDataEnd] = useState("2023-12-31");
  const [trainStart, setTrainStart] = useState("2015-01-01");
  const [trainEnd, setTrainEnd] = useState("2020-12-31");
  const [evalStart, setEvalStart] = useState("2021-01-01");
  const [evalEnd, setEvalEnd] = useState("2023-12-31");
  const [modelName, setModelName] = useState("gradient_boosting");
  const [runTag, setRunTag] = useState("");
  const [retrainLookback, setRetrainLookback] = useState("3");
  const [minRetrainRows, setMinRetrainRows] = useState("400");
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function autoTag() {
    return `${ticker}_${trainEnd.slice(0,7)}__${evalStart.slice(0,7)}_${evalEnd.slice(0,7)}`;
  }

  async function launch() {
    setError(null);
    setLaunching(true);
    const tag = runTag.trim() || autoTag();
    const cfg = {
      source, ticker_or_series: ticker, csv_path: null, date_col: null,
      schema_mode: "auto", open_col: null, high_col: null, low_col: null,
      close_col: null, volume_col: null,
      data_start: dataStart, data_end: dataEnd,
      train_start: trainStart, train_end: trainEnd,
      eval_start: evalStart, eval_end: evalEnd,
      model_name: modelName,
      retrain_lookback_years: parseFloat(retrainLookback),
      min_retrain_rows: parseInt(minRetrainRows),
      run_tag: tag,
    };
    try {
      const { run_id } = await api.createRun(cfg);
      onStartStream(run_id, tag);
    } catch (e) {
      setError(String(e));
    } finally {
      setLaunching(false);
    }
  }

  const steps = ["Data", "Dates", "Model", "Review"];

  // If stream is active, show terminal
  if (stream) {
    return (
      <div className="flex-1 flex flex-col overflow-hidden p-6 gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-[#f5f5f5]">
              {stream.done ? "Run Complete" : "Running Pipeline"}
            </h1>
            <p className="font-mono text-xs text-[#6b6b7a] mt-0.5">{stream.tag}</p>
          </div>
          <div className="flex items-center gap-4">
            {!stream.done && (
              <>
                <span className="font-mono text-sm text-[#6b6b7a]">&#9201; {fmt(stream.elapsed)}</span>
                <span className="font-mono text-sm text-[#d4a853]">{stream.progress}%</span>
              </>
            )}
            {stream.done && !stream.error && (
              <button onClick={() => onDone(stream.tag)} className="btn-primary">
                View Results &rarr;
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {!stream.done && (
          <div className="h-1.5 bg-[#2a2a2d] rounded-full overflow-hidden">
            <div
              className="h-full bg-[#d4a853] rounded-full transition-all duration-700"
              style={{ width: `${stream.progress}%` }}
            />
          </div>
        )}

        {/* Terminal */}
        <div className="flex-1 bg-[#0f0f0f] border border-[#2a2a2d] rounded-xl overflow-hidden flex flex-col">
          <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-[#2a2a2d]">
            <span className="w-2.5 h-2.5 rounded-full bg-[#3f3f46]" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#3f3f46]" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#3f3f46]" />
            <span className="ml-2 font-mono text-[10px] text-[#3f3f46]">pipeline output</span>
            {!stream.done && <span className="ml-auto font-mono text-[10px] text-[#4ade80] animate-pulse">● live</span>}
          </div>
          <div className="flex-1 overflow-y-auto p-4 font-mono text-[11px] leading-relaxed space-y-0.5">
            {stream.logs.map((line, i) => (
              <p key={i} style={{ color: logColor(line) }}>{line || "\u00a0"}</p>
            ))}
            {!stream.done && (
              <p className="text-[#3f3f46] animate-pulse">_</p>
            )}
            {stream.error && (
              <p className="text-[#f87171] mt-2">&#9888; {stream.error}</p>
            )}
            {stream.done && !stream.error && (
              <p className="text-[#4ade80] mt-2">&#10003; Pipeline completed successfully.</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <h1 className="text-xl font-semibold text-[#f5f5f5]">New Run</h1>
          <p className="text-sm text-[#6b6b7a] mt-1">Configure and launch the drift detection pipeline.</p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-0 mb-8">
          {steps.map((s, i) => (
            <div key={i} className="flex items-center flex-1">
              <button
                onClick={() => i <= step && setStep(i)}
                className={`flex items-center gap-2 ${i < step ? "cursor-pointer" : "cursor-default"}`}
              >
                <span className={`w-7 h-7 rounded-full flex items-center justify-center font-mono text-xs font-semibold transition-colors ${
                  i < step ? "bg-[#4ade80]/20 text-[#4ade80] border border-[#4ade80]/30"
                  : i === step ? "bg-[#d4a853]/20 text-[#d4a853] border border-[#d4a853]/50"
                  : "bg-[#1e1e21] text-[#3f3f46] border border-[#2a2a2d]"
                }`}>
                  {i < step ? "✓" : i + 1}
                </span>
                <span className={`text-sm ${i === step ? "text-[#f5f5f5]" : "text-[#6b6b7a]"} hidden sm:block`}>{s}</span>
              </button>
              {i < steps.length - 1 && (
                <div className={`flex-1 h-px mx-3 ${i < step ? "bg-[#4ade80]/30" : "bg-[#2a2a2d]"}`} />
              )}
            </div>
          ))}
        </div>

        {error && (
          <div className="card border-[#f87171]/30 mb-4 font-mono text-xs text-[#f87171]">&#9888; {error}</div>
        )}

        {/* Step 0: Data source */}
        {step === 0 && (
          <div className="card space-y-4">
            <p className="text-[#f5f5f5] font-semibold">Data Source</p>
            <div className="space-y-2">
              {SOURCES.map((s) => (
                <label key={s.value} className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                  source === s.value ? "border-[#d4a853]/50 bg-[#d4a853]/5" : "border-[#2a2a2d] hover:border-[#3f3f46]"
                }`}>
                  <input type="radio" name="source" value={s.value} checked={source === s.value} onChange={() => setSource(s.value)} className="mt-0.5 accent-[#d4a853]" />
                  <span className="text-sm text-[#f5f5f5]">{s.label}</span>
                </label>
              ))}
            </div>
            <div>
              <label className="label block mb-1.5">Ticker / Series</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                placeholder="e.g. ^GSPC, BTC-USD, DGS10"
                className="w-full bg-[#0f0f0f] border border-[#2a2a2d] rounded-lg px-3 py-2 font-mono text-sm text-[#f5f5f5] focus:outline-none focus:border-[#d4a853]/50 placeholder-[#3f3f46]"
              />
            </div>
          </div>
        )}

        {/* Step 1: Dates */}
        {step === 1 && (
          <div className="card space-y-4">
            <p className="text-[#f5f5f5] font-semibold">Date Ranges</p>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: "Data Start", val: dataStart, set: setDataStart },
                { label: "Data End", val: dataEnd, set: setDataEnd },
                { label: "Train Start", val: trainStart, set: setTrainStart },
                { label: "Train End", val: trainEnd, set: setTrainEnd },
                { label: "Eval Start", val: evalStart, set: setEvalStart },
                { label: "Eval End", val: evalEnd, set: setEvalEnd },
              ].map(({ label, val, set }) => (
                <div key={label}>
                  <label className="label block mb-1.5">{label}</label>
                  <input
                    type="date"
                    value={val}
                    onChange={(e) => set(e.target.value)}
                    className="w-full bg-[#0f0f0f] border border-[#2a2a2d] rounded-lg px-3 py-2 font-mono text-sm text-[#f5f5f5] focus:outline-none focus:border-[#d4a853]/50"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step 2: Model */}
        {step === 2 && (
          <div className="card space-y-4">
            <p className="text-[#f5f5f5] font-semibold">Model &amp; Adaptation</p>
            <div className="space-y-2">
              {MODELS.map((m) => (
                <label key={m.value} className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                  modelName === m.value ? "border-[#d4a853]/50 bg-[#d4a853]/5" : "border-[#2a2a2d] hover:border-[#3f3f46]"
                }`}>
                  <input type="radio" name="model" value={m.value} checked={modelName === m.value} onChange={() => setModelName(m.value)} className="accent-[#d4a853]" />
                  <span className="text-sm text-[#f5f5f5]">{m.label}</span>
                </label>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label block mb-1.5">Retrain Lookback (years)</label>
                <input
                  type="number" min="1" max="10" step="0.5"
                  value={retrainLookback}
                  onChange={(e) => setRetrainLookback(e.target.value)}
                  className="w-full bg-[#0f0f0f] border border-[#2a2a2d] rounded-lg px-3 py-2 font-mono text-sm text-[#f5f5f5] focus:outline-none focus:border-[#d4a853]/50"
                />
              </div>
              <div>
                <label className="label block mb-1.5">Min Retrain Rows</label>
                <input
                  type="number" min="100" max="2000" step="50"
                  value={minRetrainRows}
                  onChange={(e) => setMinRetrainRows(e.target.value)}
                  className="w-full bg-[#0f0f0f] border border-[#2a2a2d] rounded-lg px-3 py-2 font-mono text-sm text-[#f5f5f5] focus:outline-none focus:border-[#d4a853]/50"
                />
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Review */}
        {step === 3 && (
          <div className="card space-y-4">
            <p className="text-[#f5f5f5] font-semibold">Review &amp; Launch</p>
            <div className="grid grid-cols-2 gap-2 font-mono text-xs">
              {[
                ["Source", source],
                ["Ticker/Series", ticker],
                ["Data Range", `${dataStart} → ${dataEnd}`],
                ["Train Range", `${trainStart} → ${trainEnd}`],
                ["Eval Range", `${evalStart} → ${evalEnd}`],
                ["Model", modelName],
                ["Retrain Lookback", `${retrainLookback}yr`],
                ["Min Rows", minRetrainRows],
              ].map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="text-[#6b6b7a] w-32 shrink-0">{k}</span>
                  <span className="text-[#f5f5f5] truncate">{v}</span>
                </div>
              ))}
            </div>
            <div>
              <label className="label block mb-1.5">Run Tag (optional — auto-generated if blank)</label>
              <input
                type="text"
                value={runTag}
                onChange={(e) => setRunTag(e.target.value)}
                placeholder={autoTag()}
                className="w-full bg-[#0f0f0f] border border-[#2a2a2d] rounded-lg px-3 py-2 font-mono text-sm text-[#f5f5f5] focus:outline-none focus:border-[#d4a853]/50 placeholder-[#3f3f46]"
              />
            </div>
          </div>
        )}

        {/* Navigation buttons */}
        <div className="flex items-center justify-between mt-6">
          <button
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
            className="btn-ghost disabled:opacity-30"
          >
            &larr; Back
          </button>
          {step < steps.length - 1 ? (
            <button onClick={() => setStep((s) => s + 1)} className="btn-primary">
              Next &rarr;
            </button>
          ) : (
            <button onClick={launch} disabled={launching} className="btn-primary">
              {launching ? "Launching…" : "Launch Run →"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
