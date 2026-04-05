from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="Drift Detection Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR  = Path("results")
_PENDING_RUNS: dict[str, str] = {}   # run_id -> temp config path
_ROOT = Path(__file__).parent


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, bool):
        return obj
    return obj


def _tags():
    if not RESULTS_DIR.exists():
        return []
    tags = {f.stem.replace("equity_curves_", "") for f in RESULTS_DIR.glob("equity_curves_*.csv")}
    return sorted(tags, key=lambda t: (RESULTS_DIR / f"equity_curves_{t}.csv").stat().st_mtime, reverse=True)


@app.get("/api/runs")
def list_runs():
    result = []
    for tag in _tags():
        meta: dict = {"tag": tag}
        p = RESULTS_DIR / f"daily_monitoring_{tag}.csv"
        if p.exists():
            try:
                want = {"date", "y_true_next", "y_pred_static", "y_pred_adaptive", "drift_event", "action"}
                df = pd.read_csv(p, usecols=lambda c: c in want)
                meta["n_obs"] = len(df)
                if "drift_event" not in df.columns and "action" in df.columns:
                    df["drift_event"] = (df["action"] != "none").astype(int)
                meta["n_drift_events"] = int(df["drift_event"].sum()) if "drift_event" in df.columns else 0
                if "y_true_next" in df.columns:
                    y = df["y_true_next"].astype(int)
                    if "y_pred_static" in df.columns:
                        meta["acc_static"] = round(float((df["y_pred_static"].astype(int) == y).mean()), 4)
                    if "y_pred_adaptive" in df.columns:
                        meta["acc_adaptive"] = round(float((df["y_pred_adaptive"].astype(int) == y).mean()), 4)
                if "date" in df.columns:
                    meta["date_min"] = str(df["date"].iloc[0])
                    meta["date_max"] = str(df["date"].iloc[-1])
            except Exception:
                pass
        result.append(meta)
    return result


@app.get("/api/runs/{tag}/daily")
def get_daily(tag: str):
    path = RESULTS_DIR / f"daily_monitoring_{tag}.csv"
    if not path.exists():
        raise HTTPException(404, "Not found")
    df = pd.read_csv(path)
    return _clean(df.replace({np.nan: None}).to_dict(orient="records"))


@app.get("/api/runs/{tag}/equity")
def get_equity(tag: str):
    path = RESULTS_DIR / f"equity_curves_{tag}.csv"
    if not path.exists():
        raise HTTPException(404, "Not found")
    df = pd.read_csv(path)
    return _clean(df.replace({np.nan: None}).to_dict(orient="records"))


@app.get("/api/runs/{tag}/drift_events")
def get_drift_events(tag: str):
    path = RESULTS_DIR / f"drift_events_{tag}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return _clean(df.replace({np.nan: None}).to_dict(orient="records"))


@app.get("/api/runs/{tag}/evaluate")
def evaluate_run(tag: str):
    daily_path = RESULTS_DIR / f"daily_monitoring_{tag}.csv"
    equity_path = RESULTS_DIR / f"equity_curves_{tag}.csv"
    if not daily_path.exists() or not equity_path.exists():
        raise HTTPException(404, "Result files not found")

    from src.evaluation.run_report import per_period_breakdown, run_evaluation_from_results

    result = run_evaluation_from_results(str(daily_path), str(equity_path), print_head=False)

    try:
        daily = pd.read_csv(str(daily_path))
        eq = pd.read_csv(str(equity_path))
        pp = per_period_breakdown(daily, eq, period="YE")
        result["per_period"] = pp.replace({np.nan: None}).to_dict(orient="records")
    except Exception:
        result["per_period"] = []

    return _clean(result)


@app.get("/api/runs/{tag}/correlations")
def get_correlations(tag: str):
    path = RESULTS_DIR / f"daily_monitoring_{tag}.csv"
    if not path.exists():
        raise HTTPException(404, "Not found")
    df = pd.read_csv(path)
    skip = {"y_true_next", "y_pred_static", "y_pred_adaptive", "drift_event"}
    numeric = df.select_dtypes(include=[np.number])
    cols = [c for c in numeric.columns if c not in skip]
    if not cols:
        return {"columns": [], "matrix": []}
    sub = numeric[cols].dropna()
    corr = sub.corr(method="pearson")
    return _clean({
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    })


@app.get("/api/runs/{tag}/roc_pr")
def get_roc_pr(tag: str):
    path = RESULTS_DIR / f"daily_monitoring_{tag}.csv"
    if not path.exists():
        raise HTTPException(404, "Not found")
    df = pd.read_csv(path)
    needed = {"p1_static", "p1_adaptive", "y_true_next"}
    if not needed.issubset(df.columns):
        raise HTTPException(400, f"Columns needed: {needed}")

    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    y = df["y_true_next"].astype(int).to_numpy()
    p_s = df["p1_static"].astype(float).to_numpy()
    p_a = df["p1_adaptive"].astype(float).to_numpy()

    fpr_s, tpr_s, _ = roc_curve(y, p_s)
    fpr_a, tpr_a, _ = roc_curve(y, p_a)
    prec_s, rec_s, _ = precision_recall_curve(y, p_s)
    prec_a, rec_a, _ = precision_recall_curve(y, p_a)

    def _ds(arr: np.ndarray, n: int = 250) -> list:
        if len(arr) <= n:
            return arr.tolist()
        idx = np.linspace(0, len(arr) - 1, n, dtype=int)
        return arr[idx].tolist()

    return _clean({
        "auc_static":   round(float(auc(fpr_s, tpr_s)), 4),
        "auc_adaptive": round(float(auc(fpr_a, tpr_a)), 4),
        "roc_static":   [{"fpr": f, "tpr": t} for f, t in zip(_ds(fpr_s), _ds(tpr_s))],
        "roc_adaptive": [{"fpr": f, "tpr": t} for f, t in zip(_ds(fpr_a), _ds(tpr_a))],
        "pr_static":    [{"rec": r, "prec": p} for r, p in zip(_ds(rec_s), _ds(prec_s))],
        "pr_adaptive":  [{"rec": r, "prec": p} for r, p in zip(_ds(rec_a), _ds(prec_a))],
    })


@app.post("/api/run")
async def create_run(config: dict):
    run_id = uuid.uuid4().hex[:10]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="drift_cfg_", dir=str(_ROOT)
    ) as fh:
        json.dump(config, fh)
        _PENDING_RUNS[run_id] = fh.name
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    cfg_path = _PENDING_RUNS.get(run_id)
    if not cfg_path:
        raise HTTPException(404, "Run not found or already consumed")

    async def _gen():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "main.py", "--json-config", cfg_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_ROOT),
        )
        assert proc.stdout
        async for raw in proc.stdout:
            text = raw.decode("utf-8", errors="replace").rstrip("\n\r")
            yield f"data: {text}\n\n"
        await proc.wait()
        try:
            Path(cfg_path).unlink(missing_ok=True)
        except Exception:
            pass
        _PENDING_RUNS.pop(run_id, None)
        yield "data: __DONE__\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/runs/{tag}")
def delete_run(tag: str):
    tag = tag  # FastAPI path param is already decoded
    deleted = []
    for name in [
        f"daily_monitoring_{tag}.csv",
        f"equity_curves_{tag}.csv",
        f"drift_events_{tag}.csv",
        f"rolling_curves_{tag}.csv",
    ]:
        p = RESULTS_DIR / name
        if p.exists():
            p.unlink()
            deleted.append(name)
    for p in RESULTS_DIR.glob(f"chart_*_{tag}.png"):
        p.unlink()
        deleted.append(p.name)
    if not deleted:
        raise HTTPException(404, "Run not found")
    return {"deleted": deleted, "tag": tag}


_frontend = Path("frontend/dist")
if _frontend.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def _spa(full_path: str):
        return FileResponse(str(_frontend / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
