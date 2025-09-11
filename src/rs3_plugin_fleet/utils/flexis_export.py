# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Helpers temps absolu/relatif
# ------------------------------------------------------------
def _compute_abs_time(df: pd.DataFrame, sim_start_utc) -> pd.Series:
    """
    Calcule l'heure absolue (UTC) à partir des colonnes disponibles en se
    protégeant contre l'ambiguïté "epoch vs temps relatif".
    Priorité (avec heuristiques):
      1) timestamp (datetime absolu crédible)
      2) ts_ms / ts_s interprétés comme EPOCH *si* leur ordre de grandeur est crédible
      3) t_ms / t_s / t_abs_s interprétés comme temps relatif (ajoutés à sim_start)
    Si aucune colonne exploitable n'est trouvée, retourne une Series de NaT.
    Les deltas relatifs sont clampés à ±100 ans pour éviter OutOfBoundsTimedelta.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    # sim_start robuste
    try:
        sim_start = pd.to_datetime(sim_start_utc, utc=True, errors="coerce")
    except Exception:
        sim_start = None
    if pd.isna(sim_start):
        sim_start = pd.Timestamp.now(tz="UTC")

    # Helpers
    def _as_numeric(s, default_nan=True):
        x = pd.to_numeric(s, errors="coerce")
        if default_nan:
            return x
        return x.fillna(0)

    # 1) 'timestamp' si crédible (datetime absolu, pas 1970 par erreur)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            years = ts.dt.year.dropna()
            if not (len(years) > 0 and (years.le(1971).all())):
                return ts.dt.tz_convert("UTC")

    # 2) Epoch: ts_ms ou ts_s, mais seulement si l'ordre de grandeur est plausible
    if "ts_ms" in df.columns:
        ts_ms_num = _as_numeric(df["ts_ms"])
        q = ts_ms_num.quantile(0.9)
        if pd.notna(q) and q >= 1e11:
            return pd.to_datetime(ts_ms_num, unit="ms", utc=True, errors="coerce").dt.tz_convert("UTC")

    if "ts_s" in df.columns:
        ts_s_num = _as_numeric(df["ts_s"])
        q = ts_s_num.quantile(0.9)
        if pd.notna(q) and q >= 1e9:
            return pd.to_datetime(ts_s_num, unit="s", utc=True, errors="coerce").dt.tz_convert("UTC")

    # 3) Relatif → sim_start + delta
    clip_win_ms = int(100 * 365.25 * 24 * 3600 * 1000)

    if "t_ms" in df.columns:
        t_ms = _as_numeric(df["t_ms"]).fillna(0).clip(-clip_win_ms, clip_win_ms).astype("int64")
        dt = pd.to_timedelta(t_ms, unit="ms")
        return (pd.Series(sim_start, index=df.index) + dt).dt.tz_convert("UTC")

    if "t_s" in df.columns:
        t_s = _as_numeric(df["t_s"]).fillna(0)
        t_ms = (t_s * 1000.0).round().clip(-clip_win_ms, clip_win_ms).astype("int64")
        dt = pd.to_timedelta(t_ms, unit="ms")
        return (pd.Series(sim_start, index=df.index) + dt).dt.tz_convert("UTC")

    if "t_abs_s" in df.columns:
        t_abs_s = _as_numeric(df["t_abs_s"]).fillna(0)
        q = t_abs_s.quantile(0.9)
        if pd.notna(q) and q >= 1e9:
            return pd.to_datetime(t_abs_s, unit="s", utc=True, errors="coerce").dt.tz_convert("UTC")
        t_ms = (t_abs_s * 1000.0).round().clip(-clip_win_ms, clip_win_ms).astype("int64")
        dt = pd.to_timedelta(t_ms, unit="ms")
        return (pd.Series(sim_start, index=df.index) + dt).dt.tz_convert("UTC")

    # 4) Rien de disponible → NaT
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

# ------------------------------------------------------------
# Trim fin de trace (arrêt prolongé)
# ------------------------------------------------------------
def _trim_tail_idle(df: pd.DataFrame, tail_window_s: float = 5.0) -> pd.DataFrame:
    """
    Supprime la traîne terminale "à l'arrêt" (accroche carte/arrêt prolongé).
    """
    if df is None or len(df) == 0 or "t_ms" not in df.columns:
        return df

    v = None
    for c in ("speed_mps", "speed", "v"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            break
    if v is None:
        v = pd.Series(np.nan, index=df.index)

    moved = pd.Series(False, index=df.index)
    if {"lat", "lon"}.issubset(df.columns):
        dlat = pd.to_numeric(df["lat"], errors="coerce").diff().abs()
        dlon = pd.to_numeric(df["lon"], errors="coerce").diff().abs()
        moved = ((dlat > 1e-5) | (dlon > 1e-5)).fillna(False)

    active = (v > 0.2) | moved
    if active.any():
        last_active_idx = active[active].index[-1]
    else:
        return df

    t_ms = pd.to_numeric(df["t_ms"], errors="coerce").ffill().fillna(0)
    tail_len_ms = t_ms.iloc[-1] - t_ms.loc[last_active_idx]
    if tail_len_ms > (tail_window_s * 1000.0):
        return df.loc[:last_active_idx]
    return df

# ------------------------------------------------------------
# Résolution robuste de outdir / run_name depuis le ctx
# ------------------------------------------------------------
def _resolve_outdir(ctx, default_dir: str) -> str:
    """
    Essaie diverses conventions pour retrouver le même outdir que le core/rapports.
    """
    for attr in ("outdir", "output_dir", "out_dir"):
        val = getattr(ctx, attr, None)
        if isinstance(val, (str, Path)) and str(val).strip():
            return str(val)

    output = getattr(ctx, "output", None)
    if output is not None:
        if isinstance(output, dict):
            val = output.get("dir")
            if isinstance(val, (str, Path)) and str(val).strip():
                return str(val)
        else:
            val = getattr(output, "dir", None)
            if isinstance(val, (str, Path)) and str(val).strip():
                return str(val)

    config = getattr(ctx, "config", None)
    if isinstance(config, dict):
        out = config.get("output") or {}
        val = out.get("dir") or config.get("outdir")
        if isinstance(val, (str, Path)) and str(val).strip():
            return str(val)

    return default_dir

def _resolve_run_name(ctx, default_filename: str) -> str:
    """
    Retrouve la même base de nom que le core.
    """
    for attr in ("name", "run_name"):
        val = getattr(ctx, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    config = getattr(ctx, "config", None)
    if isinstance(config, dict):
        for key in ("name", "run_name"):
            val = config.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        out = config.get("output") or {}
        val = out.get("name")
        if isinstance(val, str) and val.strip():
            return val.strip()

    base = os.path.splitext(default_filename)[0] or "run"
    return base

# ------------------------------------------------------------
# Exporter
# ------------------------------------------------------------
@dataclass
class Config:
    out_dir: str
    filename: str = "flexis.csv"
    write_csv: bool = True
    write_parquet: bool = False
    write_events_jsonl: bool = False
    tail_window_s: float = 5.0

class Stage:
    """Exporter Flexis : écrit CSV/Parquet + colonnes temps absolu (time_utc, ts_ms)."""
    name = "FlexisExporter"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        cfg = cfg or {}
        export = (cfg.get("export") or {})
        out_dir = (cfg.get("output", {}) or {}).get("dir") or export.get("dir") or "data/simulations/default"
        filename = export.get("filename", "flexis.csv")
        tail_window = float(export.get("tail_window", 5))
        self.cfg = Config(out_dir=out_dir, filename=filename, tail_window_s=tail_window)
        self.last_out_path: Optional[str] = None

    def process(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        df = df.copy()

        # 1) Recyclage si déjà présent, sinon calcul robuste
        if "time_utc" in df.columns:
            abs_time = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
        else:
            sim_start = getattr(ctx, "sim_start", None) or pd.Timestamp.now(tz="UTC")
            abs_time = _compute_abs_time(df, sim_start)

        # Normaliser les colonnes dérivées cohérentes
        df["time_utc"] = abs_time
        df["ts_ms"] = (abs_time.astype("int64") // 1_000_000).astype("int64")
        df["timestamp"] = abs_time

        # 2) trim de la queue "à l'arrêt"
        df = _trim_tail_idle(df, self.cfg.tail_window_s)

        # 3) Écriture alignée avec le core/rapports
        out_dir = _resolve_outdir(ctx, self.cfg.out_dir)
        run_name = _resolve_run_name(ctx, self.cfg.filename)

        # Évite le double suffixe si run_name termine déjà par '-flexis'
        out_basename = run_name if run_name.lower().endswith("-flexis") else f"{run_name}-flexis"
        out_path = Path(out_dir) / f"{out_basename}.csv"
        self.last_out_path = str(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Build an explicit export dataframe that guarantees all flexis_* columns are kept ---
        base_cols = [
            "time_utc", "ts_ms", "timestamp",  # absolute time columns created above
            "t_ms", "t_s", "t_abs_s",          # relative time if present
            "lat", "lon", "speed_mps", "road_type", "is_service",
        ]

        # Ensure a fixed canonical Flexis schema is present (create empty defaults when missing)
        required_flexis = [
            "flexis_road_type",
            "flexis_road_curve_radius_m",
            "flexis_nav_event",
            "flexis_infra_event",
            "flexis_night",
            "flexis_driver_event",
            "flexis_traffic_level",
            "flexis_weather",
            "flexis_delivery_status",
            "flexis_population_density_km2",
        ]

        # Force l'ajout des colonnes flexis_* même si elles sont manquantes dans df
        for col in required_flexis:
            if col not in df.columns:
                if col == "flexis_night":
                    df[col] = False
                elif col in ("flexis_road_curve_radius_m", "flexis_population_density_km2"):
                    df[col] = np.nan
                else:
                    # semantic default to prevent blank columns in prints/exports
                    default = "unknown" if col not in ("flexis_nav_event", "flexis_infra_event", "flexis_driver_event") else ""
                    df[col] = default

        # --- Normalize flexis_* BEFORE debug to avoid empty-looking heads ---
        flexis_cols_all = [c for c in df.columns if isinstance(c, str) and c.startswith("flexis_")]
        for col in flexis_cols_all:
            s = df[col]
            if pd.api.types.is_object_dtype(s.dtype):
                s2 = s.astype(str).replace({"nan": "", "None": ""}).str.strip()
                default = "unknown" if col not in ("flexis_nav_event", "flexis_infra_event", "flexis_driver_event") else ""
                s2 = s2.replace("", default)
                df[col] = s2
            else:
                if col == "flexis_population_density_km2":
                    df[col] = pd.to_numeric(s, errors="coerce").fillna(600.0)
                elif col == "flexis_road_curve_radius_m":
                    s_num = pd.to_numeric(s, errors="coerce")
                    df[col] = s_num.bfill().ffill().fillna(5000.0)
                else:
                    df[col] = pd.to_numeric(s, errors="coerce")

        # Debug: afficher les valeurs des colonnes flexis_*
        print("[DEBUG] Colonnes flexis_* avant export:")
        for col in required_flexis:
            if col in df.columns:
                print(f"  {col}: {df[col].head()}")

        # Collect ALL columns, ensuring flexis_* are included
        all_cols = list(df.columns)
        flexis_cols = [c for c in all_cols if isinstance(c, str) and c.startswith("flexis_")]

        # Deduplicate while preserving order: base_cols first, then flexis_cols, then the rest
        cols_order = []
        for c in base_cols:
            if c in all_cols and c not in cols_order:
                cols_order.append(c)
        for c in flexis_cols:
            if c not in cols_order:
                cols_order.append(c)
        for c in all_cols:
            if c not in cols_order:
                cols_order.append(c)

        df_out = df[cols_order].copy()

        # Normalize types for a few columns
        if "flexis_night" in df_out.columns:
            try:
                df_out["flexis_night"] = df_out["flexis_night"].astype(bool)
            except Exception:
                df_out["flexis_night"] = df_out["flexis_night"].fillna(False)

        # Normalize flexis_* values and dtypes for robust export
        for col in flexis_cols:
            if col not in df_out.columns:
                continue
            s = df_out[col]
            if pd.api.types.is_object_dtype(s.dtype):
                # Clean strings, replace None/nan textual artifacts, and trim
                s = s.astype(str).replace({"nan": "", "None": ""}).str.strip()
                # For event columns we allow empty strings, else default to 'unknown'
                default = "unknown" if col not in ("flexis_nav_event", "flexis_infra_event", "flexis_driver_event") else ""
                s = s.replace("", default)
                df_out[col] = s
            else:
                if col == "flexis_population_density_km2":
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(600.0)
                elif col == "flexis_road_curve_radius_m":
                    s_num = pd.to_numeric(df_out[col], errors="coerce")
                    df_out[col] = s_num.bfill().ffill().fillna(5000.0)
                else:
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

        # write the curated dataframe according to config flags
        wrote_paths = []
        want_parquet = bool(self.cfg.write_parquet) or str(out_path).lower().endswith(".parquet")
        want_csv = bool(self.cfg.write_csv) or not want_parquet

        if want_parquet:
            pq_path = out_path.with_suffix(".parquet")
            df_out.to_parquet(pq_path, index=False)
            wrote_paths.append(str(pq_path))

        if want_csv:
            csv_path = out_path if out_path.suffix.lower() == ".csv" else out_path.with_suffix(".csv")
            df_out.to_csv(csv_path, index=False)
            wrote_paths.append(str(csv_path))

        print(f"[FlexisExporter] Wrote {', '.join(wrote_paths)}")

        # 4) remettre dans le ctx (chaînage éventuel)
        try:
            setattr(ctx, "df", df_out)
        except Exception:
            pass

        return df_out

    def run(self, ctx):
        """Core2 entrypoint: consume ctx.df (or ctx.timeline) and write exports, returning a status dict."""
        df = getattr(ctx, "df", None)
        if df is None:
            df = getattr(ctx, "timeline", None)
        if df is None:
            return {"ok": False, "msg": "flexis_export.Stage.run: no dataframe found on ctx (expected ctx.df or ctx.timeline)"}

        try:
            self.process(df, ctx)
        except Exception as e:
            return {"ok": False, "msg": f"flexis_export.Stage.run failed: {e}"}

        msg = f"Wrote {self.last_out_path}" if self.last_out_path else "flexis export written"
        return {"ok": True, "msg": msg}
