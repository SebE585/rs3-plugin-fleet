# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict
import re
import pandas as pd


# ---------- helpers -----------------------------------------------------------
def slugify(s: str) -> str:
    if not s:
        return "default"
    s = re.sub(r"[^0-9A-Za-z._-]+", "-", str(s).strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "default"


# ---------- Robust UTC timestamp parser --------------------------------------
def parse_ts_utc(ts: pd.Series) -> pd.Series:
    """
    Parse une série hétérogène en datetimes UTC (epoch s/ms/ns, datetimes, strings).
    """
    if ts is None:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    # Essai numérique (epoch s/ms/ns)
    s = pd.to_numeric(ts, errors="coerce")
    if s.notna().any():
        mx = s.dropna().abs().max()
        try:
            if mx < 1e11:      # secondes
                return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            elif mx < 1e14:    # millisecondes
                return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            else:              # nanosecondes
                return pd.to_datetime(s, unit="ns", utc=True, errors="coerce")
        except Exception:
            pass

    # Sinon : laisser pandas parser (datetimes / strings)
    return pd.to_datetime(ts, utc=True, errors="coerce")


def tz_aware_utc_from_config(cfg: Dict[str, Any]) -> pd.Timestamp:
    """
    Calcule le start_at (UTC) à partir de la config (accepte start_at top-level,
    ou dans simulation/fleet), avec fallback 08:00 Europe/Paris aujourd'hui.
    """
    start_at = (
        cfg.get("start_at")
        or cfg.get("simulation", {}).get("start_at")
        or cfg.get("fleet", {}).get("start_at")
    )
    if start_at:
        ts = pd.Timestamp(start_at)
        if ts.tz is None:
            ts = ts.tz_localize("Europe/Paris").tz_convert("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    # fallback : aujourd’hui 08:00 Europe/Paris → UTC
    ts_local = pd.Timestamp(pd.Timestamp.now(tz="Europe/Paris").date()).replace(
        hour=8, minute=0, second=0, microsecond=0
    )
    return ts_local.tz_convert("UTC")


def ensure_sim_start_on_ctx(ctx: Any, cfg: Dict[str, Any]) -> pd.Timestamp:
    """
    Consolide ctx.sim_start → tz-aware UTC (utilisé par core et export Flexis).
    """
    sim_start = getattr(ctx, "sim_start", None)
    if sim_start is None:
        sim_start = tz_aware_utc_from_config(cfg)
        try:
            setattr(ctx, "sim_start", sim_start)
        except Exception:
            pass
    else:
        ts = pd.Timestamp(sim_start)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        setattr(ctx, "sim_start", ts)
        sim_start = ts
    return sim_start


def ensure_export_defaults(run_cfg: Dict[str, Any]) -> None:
    """
    Garantit output/export/start_at pour éviter 1970 et assurer les sorties.
    """
    # 1) output dir: on dérive d'abord du nom/client sinon fallback 'default'
    base_name = (
        run_cfg.get("output", {}).get("dir_name")
        or run_cfg.get("name")
        or run_cfg.get("client")
        or "default"
    )
    explicit_dir = (
        run_cfg.get("output", {}).get("dir")
        or run_cfg.get("export", {}).get("dir")
        or run_cfg.get("outdir")
    )
    out_dir = explicit_dir or f"data/simulations/{slugify(base_name)}"
    run_cfg.setdefault("output", {})
    run_cfg["output"]["dir"] = out_dir

    # 2) export defaults
    exp = run_cfg.setdefault("export", {})
    exp.setdefault("format", "parquet")
    exp.setdefault("filename", "flexis.parquet")
    exp.setdefault("events_filename", "events.jsonl")
    exp.setdefault("close_tail_stop", True)
    exp.setdefault("tail_window", 5)  # secondes

    # 3) start_at si absent → 08:00 Europe/Paris aujourd'hui
    if "start_at" not in run_cfg:
        ts_local = pd.Timestamp(pd.Timestamp.now(tz="Europe/Paris").date())
        run_cfg["start_at"] = ts_local.replace(hour=8, minute=0, second=0).strftime("%Y-%m-%d %H:%M")


def ensure_relative_time_columns(df: pd.DataFrame, ctx: Any) -> pd.DataFrame:
    """
    Reconstruit t_ms/t_s si absents à partir de timestamp/time_utc/ts_ms + ctx.sim_start.
    Clamp ±100 ans pour éviter OutOfBoundsTimedelta.
    """
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    if ("t_ms" in df.columns) or ("t_s" in df.columns):
        return df

    sim_start = getattr(ctx, "sim_start", None) or pd.Timestamp.now(tz="UTC")
    sim_start = pd.Timestamp(sim_start).tz_convert("UTC")
    start_ms = int(sim_start.value // 1_000_000)

    # ts_ms absolu → relatif
    if "ts_ms" in df.columns:
        ts_raw = pd.to_numeric(df["ts_ms"], errors="coerce")
        if ts_raw.notna().any():
            mx = ts_raw.dropna().abs().max()
            if mx < 1e11:   # s
                ts_ms = ts_raw * 1000.0
            elif mx < 1e14: # ms
                ts_ms = ts_raw
            else:           # ns
                ts_ms = ts_raw / 1e6
            t_ms = ts_ms - start_ms
            clip_win = int(100 * 365.25 * 24 * 3600 * 1000)
            df2 = df.copy()
            df2["t_ms"] = pd.to_numeric(t_ms, errors="coerce").fillna(0).clip(-clip_win, clip_win).astype("int64")
            return df2

    # sinon timestamp/time_utc → relatif
    ts_abs = None
    if "timestamp" in df.columns:
        ts_abs = parse_ts_utc(df["timestamp"])
    elif "time_utc" in df.columns:
        ts_abs = parse_ts_utc(df["time_utc"])
    if ts_abs is not None:
        valid = ts_abs.notna()
        ts_ms = pd.Series(0, index=df.index, dtype="int64")
        if valid.any():
            ts_ms.loc[valid] = (ts_abs[valid].astype("int64") // 1_000_000).astype("int64")
        delta_ms = ts_ms - start_ms
        clip_win = int(100 * 365.25 * 24 * 3600 * 1000)
        df2 = df.copy()
        df2["t_ms"] = pd.to_numeric(delta_ms, errors="coerce").fillna(0).clip(-clip_win, clip_win).astype("int64")
        return df2

    return df


def trim_tail_stop(df: pd.DataFrame, ctx: Any, run_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Supprime un arrêt artificiel prolongé en fin de trace.
    Utilise t_ms si dispo, sinon tente timestamp/time_utc.
    """
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()

    export_cfg = (run_cfg.get("export") or {})
    if not export_cfg.get("close_tail_stop", True):
        return df
    tail_s = int(export_cfg.get("tail_window", 5))

    # temps relatif t_ms
    tdf = ensure_relative_time_columns(df, ctx)
    if "t_ms" not in tdf.columns:
        return df

    # colonne vitesse
    v = None
    for c in ("speed_mps", "speed", "v"):
        if c in tdf.columns:
            v = pd.to_numeric(tdf[c], errors="coerce"); break
    if v is None:
        return df

    # cherche la dernière séquence >= tail_s avec v ~ 0
    t_ms = pd.to_numeric(tdf["t_ms"], errors="coerce").ffill().fillna(0).astype("int64")
    is_stop = (v.fillna(0) < 0.05)
    if not is_stop.any():
        return df

    last_t = int(t_ms.iloc[-1])
    win_start = last_t - tail_s * 1000
    tail_mask = (t_ms >= win_start)
    if (is_stop & tail_mask).sum() >= 2:
        before_tail = t_ms < win_start
        if before_tail.any():
            cut_idx = before_tail[before_tail].index[-1]
            return tdf.loc[:cut_idx].copy()
    return df