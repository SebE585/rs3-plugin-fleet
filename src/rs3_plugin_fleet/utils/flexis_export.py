# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import re
from rs3_plugin_fleet.flexis.enricher import FlexisEnricher, _extract_schedules_from_ctx

logger = logging.getLogger(__name__)

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
    Fallback temps:
      - t_ms si présent,
      - sinon ts_ms (epoch ms) → relatif,
      - sinon time_utc → relatif.
    """
    if df is None or len(df) == 0:
        return df

    # Construire une base temps en millisecondes relative au début
    t_ms = None
    if "t_ms" in df.columns:
        t_ms = pd.to_numeric(df["t_ms"], errors="coerce").ffill().fillna(0)
    elif "ts_ms" in df.columns:
        _ts = pd.to_numeric(df["ts_ms"], errors="coerce").ffill()
        if len(_ts) == 0 or _ts.iloc[0] is np.nan:
            return df
        t_ms = (_ts - _ts.iloc[0]).clip(lower=0)
    elif "time_utc" in df.columns:
        _tu = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
        _n = _tu.view("int64") // 1_000_000  # ns → ms
        if len(_n) == 0 or pd.isna(_n.iloc[0]):
            return df
        t_ms = (_n - int(_n.iloc[0])).clip(lower=0)
    else:
        # Pas de base temps exploitable
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
# Multi-véhicule helpers
# ------------------------------------------------------------
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _sanitize_segment(x: str) -> str:
    """Sanitize a string so it is safe as a folder/file segment."""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return "vehicle"
    x = _SANITIZE_RE.sub("-", x)
    return x.strip("-._") or "vehicle"


def _resolve_vehicle_id(ctx, df: pd.DataFrame | None = None) -> Optional[str]:
    """
    Tente de retrouver un identifiant véhicule depuis le contexte ou le dataframe.
    Ordre d'essai :
      - ctx.vehicle_id, ctx.vehicle.id, ctx.vehicle.name, ctx.vehicle
      - ctx.config["vehicle_id"]
      - première valeur non nulle dans df["vehicle_id"] (si dispo)
    Retourne une chaîne nettoyée pour usage dans un dossier, ou None si introuvable.
    """
    # 1) Contexte direct
    for attr in ("vehicle_id",):
        v = getattr(ctx, attr, None)
        if isinstance(v, (str, int)):  # accept simple scalar
            v = str(v).strip()
            if v:
                return _sanitize_segment(v)

    # 2) ctx.vehicle.*
    veh = getattr(ctx, "vehicle", None)
    if veh is not None:
        for attr in ("id", "name"):
            v = getattr(veh, attr, None)
            if isinstance(v, (str, int)):
                v = str(v).strip()
                if v:
                    return _sanitize_segment(v)
        # veh peut être directement une chaîne
        if isinstance(veh, str) and veh.strip():
            return _sanitize_segment(veh)

    # 3) ctx.config
    cfg = getattr(ctx, "config", None)
    if isinstance(cfg, dict):
        v = cfg.get("vehicle_id") or (cfg.get("vehicle") or {}).get("id") or (cfg.get("vehicle") or {}).get("name")
        if isinstance(v, (str, int)):
            v = str(v).strip()
            if v:
                return _sanitize_segment(v)

    # 4) Dataframe
    if df is not None and isinstance(df, pd.DataFrame) and "vehicle_id" in df.columns:
        try:
            s = df["vehicle_id"].astype(str).str.strip()
            cand = s[s != ""].iloc[0]
            if isinstance(cand, str) and cand:
                return _sanitize_segment(cand)
        except Exception:
            pass

    return None

# ------------------------------------------------------------
# Client/project slug helper
# ------------------------------------------------------------
def _resolve_client_slug(ctx) -> str:
    """Find the client/project name from ctx/cfg and return a sanitized slug.
    Tries several conventional locations in ctx and ctx.config/cfg.
    Also tries to infer from a config/yaml path if present.
    Defaults to 'default' if not found.
    """
    cand = None

    # 1) Search in ctx.cfg / ctx.config common keys
    for attr in ("cfg", "config"):
        cfg = getattr(ctx, attr, None)
        if isinstance(cfg, dict):
            # direct keys commonly used to store a human label
            for key in ("client", "project", "customer", "title", "name", "label"):
                v = cfg.get(key)
                if isinstance(v, str) and v.strip():
                    cand = v.strip()
                    break
            if cand:
                break
            # nested common containers
            for parent in ("meta", "info", "output", "sim", "dataset"):
                sub = cfg.get(parent) or {}
                if isinstance(sub, dict):
                    for key in ("client", "project", "customer", "title", "name", "label"):
                        v = sub.get(key)
                        if isinstance(v, str) and v.strip():
                            cand = v.strip()
                            break
                if cand:
                    break
            if cand:
                break

    # 2) fallbacks from ctx attributes
    if not cand:
        for attr in ("client", "project", "title", "name", "run_name"):
            v = getattr(ctx, attr, None)
            if isinstance(v, str) and v.strip():
                cand = v.strip()
                break

    # 3) last chance: try to infer from config path if present on ctx/cfg
    if not cand:
        # Try various attributes that may hold the yaml path
        paths = []
        for attr in ("config_path", "cfg_path", "yaml_path", "config_file", "cfg_file"):
            v = getattr(ctx, attr, None)
            if isinstance(v, str) and v.strip():
                paths.append(v)
        for attr in ("cfg", "config"):
            cfg = getattr(ctx, attr, None)
            if isinstance(cfg, dict):
                for k in ("config_path", "cfg_path", "yaml_path", "config_file", "cfg_file", "__cfg_path__", "__config_path__"):
                    v = cfg.get(k)
                    if isinstance(v, str) and v.strip():
                        paths.append(v)
        # derive name from the stem of the first existing path-like string
        for p in paths:
            try:
                stem = Path(p).stem  # e.g. coin-coin-delivery
                if stem:
                    cand = stem.replace("_", " ").replace("-", " ")
                    break
            except Exception:
                continue

    if not cand:
        # as a very last resort, use ctx.name/run_name if present
        for attr in ("name", "run_name"):
            v = getattr(ctx, attr, None)
            if isinstance(v, str) and v.strip():
                cand = v.strip()
                break

    return _sanitize_segment(cand) if cand else "default"

# ------------------------------------------------------------
# Helpers flexis fallback/merge
# ------------------------------------------------------------

def _is_blank_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([True] * 0)
    so = s.astype(object)
    st = so.where(~so.isna(), "").astype(str).str.strip().str.lower()
    return st.isna() | (st == "") | (st == "unknown") | (st == "nan") | (st == "none")


def _mostly_empty(df: pd.DataFrame, col: str, ratio: float = 0.98) -> bool:
    if col not in df.columns:
        return True
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s.isna().mean() >= ratio
    return float(_is_blank_series(s).mean()) >= ratio


def _merge_preferring_existing(existing: pd.Series, enriched: pd.Series, is_bool: bool = False) -> pd.Series:
    if existing is None or len(existing) == 0:
        return enriched
    if pd.api.types.is_bool_dtype(existing) or is_bool:
        ex = pd.to_numeric(existing, errors="coerce")
        enr = pd.to_numeric(enriched, errors="coerce") if enriched is not None else None
        out = ex.where(~ex.isna(), enr)
        return out.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(existing):
        ex = pd.to_numeric(existing, errors="coerce")
        if enriched is None:
            return ex
        enr = pd.to_numeric(enriched, errors="coerce")
        return ex.fillna(enr)
    ex = existing.astype(object)
    ex_blank = _is_blank_series(ex)
    if enriched is None:
        return ex
    enr = enriched.astype(object).where(~enriched.isna(), "").astype(str).str.strip()
    out = ex.copy()
    out.loc[ex_blank] = enr.loc[ex_blank]
    return out


def _ensure_flexis_columns(df: pd.DataFrame, ctx) -> pd.DataFrame:
    """
    Si des colonnes flexis_* manquent ou sont (presque) vides, on relance un enrich 
    minimal en récupérant `traffic_profile`/`weather_timeline` depuis le contexte,
    puis on fusionne en NE REMPLAÇANT QUE LES VIDES.

    Tweaks:
      - Ne considère plus `flexis_infra_event` comme motif de fallback si elle est quasi vide
        (cette colonne est souvent vide par nature).
      - Journalise colonne par colonne combien de valeurs ont été comblées.
      - Si le fallback ne change rien, log silencieux (pas de message bruyant).
    """
    flexis_cols = [
        "flexis_road_type",
        "flexis_road_curve_radius_m",
        "flexis_nav_event",
        "flexis_weather",
        "flexis_night",
        "flexis_traffic_level",
        "flexis_delivery_status",
        "flexis_infra_event",
        "flexis_driver_event",
        "flexis_population_density_km2",
    ]

    # Colonnes qu'on NE DOIT PAS considérer comme motif de fallback si quasi vides
    _non_critical_mostly_empty = {"flexis_infra_event"}

    missing = [c for c in flexis_cols if c not in df.columns]

    # Calcule les colonnes quasi vides mais exclut les non-critiques
    mostly_empty_all = [c for c in flexis_cols if c in df.columns and _mostly_empty(df, c, ratio=0.98)]
    mostly_empty = [c for c in mostly_empty_all if c not in _non_critical_mostly_empty]

    if missing or mostly_empty:
        any_change = False
        per_col_filled = {}
        try:
            logger.debug(f"[FlexisExporter] Fallback check: missing={missing} mostly_empty={mostly_empty}")
        except Exception:
            pass

        # État initial des colonnes pour le comptage des remplissages
        before_map = {}
        for col in flexis_cols:
            if col in df.columns:
                s = df[col]
                if pd.api.types.is_object_dtype(s.dtype):
                    before_map[col] = s.astype(object).where(~s.isna(), "").astype(str).str.strip()
                else:
                    before_map[col] = s

        # Récupérer les plannings depuis le contexte (mêmes règles que l'enricher)
        tp, wt = _extract_schedules_from_ctx(ctx)
        fx = FlexisEnricher({
            "traffic_profile": tp or [],
            "weather_timeline": wt or [],
            "export_after_enrich": False,
        })
        fx_df = fx.apply(df.copy(), context=ctx)

        # Fusion non destructive + comptage des comblements
        for col in flexis_cols:
            if col in fx_df.columns:
                is_bool = (col == "flexis_night")
                if col in df.columns:
                    before = before_map.get(col, df[col])
                    new_series = _merge_preferring_existing(df[col], fx_df[col], is_bool=is_bool)
                    df[col] = new_series

                    # Comptage: nombre de cellules auparavant vides/comme inconnues et maintenant remplies
                    filled_count = 0
                    try:
                        if pd.api.types.is_object_dtype(before.dtype):
                            was_blank = before.astype(object).where(~before.isna(), "").astype(str).str.strip()
                            now = new_series.astype(object).where(~new_series.isna(), "").astype(str).str.strip()
                            filled_count = int(((was_blank == "") & (now != "")).sum())
                        else:
                            # numérique/bool: rempli si était NaN et devient non-NaN
                            was_nan = pd.to_numeric(before, errors="coerce").isna()
                            now_nan = pd.to_numeric(new_series, errors="coerce").isna()
                            filled_count = int((was_nan & ~now_nan).sum())
                    except Exception:
                        filled_count = 0
                    per_col_filled[col] = filled_count
                    any_change = any_change or (filled_count > 0)
                else:
                    df[col] = fx_df[col]
                    per_col_filled[col] = int(len(df[col]))  # colonne entièrement ajoutée
                    any_change = True

        # Normalisation finale
        for col in ("flexis_road_type", "flexis_traffic_level", "flexis_weather", "flexis_delivery_status"):
            if col in df.columns:
                s = df[col].astype(object).where(~df[col].isna(), "").astype(str).str.strip()
                s = s.replace({"nan": "", "None": ""})
                df[col] = s.replace("", "unknown")

        for col in ("flexis_nav_event", "flexis_infra_event", "flexis_driver_event"):
            if col in df.columns:
                df[col] = df[col].astype(object).where(~df[col].isna(), "").astype(str).str.strip()

        if "flexis_population_density_km2" in df.columns:
            df["flexis_population_density_km2"] = pd.to_numeric(df["flexis_population_density_km2"], errors="coerce").fillna(600.0)
        if "flexis_road_curve_radius_m" in df.columns:
            s = pd.to_numeric(df["flexis_road_curve_radius_m"], errors="coerce")
            df["flexis_road_curve_radius_m"] = s.bfill().ffill().fillna(5000.0)

        # Logging final: seulement si quelque chose a réellement été comblé/ajouté
        if any_change:
            try:
                details = ", ".join([f"{k}:{v}" for k, v in per_col_filled.items() if v > 0]) or "no per-column deltas"
                logger.debug(f"[FlexisExporter] Fallback enrich applied (filled cells → {details})")
            except Exception:
                pass
        else:
            # Silence si aucun changement (évite le bruit)
            pass
    else:
        try:
            logger.debug("[FlexisExporter] Fallback not needed: flexis_* columns already populated.")
        except Exception:
            pass

    return df

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

        # Synthétiser t_ms si absent (relatif au premier échantillon)
        if "t_ms" not in df.columns or pd.to_numeric(df["t_ms"], errors="coerce").isna().all():
            _ts0 = int(df["ts_ms"].iloc[0]) if len(df["ts_ms"]) else 0
            df["t_ms"] = (pd.to_numeric(df["ts_ms"], errors="coerce") - _ts0).clip(lower=0)

        # 2) trim de la queue "à l'arrêt"
        df = _trim_tail_idle(df, self.cfg.tail_window_s)

        # 3) Écriture dans le répertoire client demandé :
        #    data/simulations/<client_slug>/<vehicle_id>.<ext>
        client_slug = _resolve_client_slug(ctx)
        out_dir_path = Path("data") / "simulations" / client_slug

        # Determine vehicle id (required for explicit file naming)
        vid = _resolve_vehicle_id(ctx, df) or _resolve_run_name(ctx, "run")

        # Extension policy: take from configured filename, else .csv
        cfg_ext = Path(self.cfg.filename).suffix
        ext = cfg_ext if cfg_ext else ".csv"

        out_path = out_dir_path / f"{vid}{ext}"
        self.last_out_path = str(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Also prepare a mirror path into the core outdir (simulated_YYYYmmdd_HHMMSS) if present
        core_outdir = _resolve_outdir(ctx, default_dir=str(out_dir_path))
        try:
            core_outdir_path = Path(core_outdir)
        except Exception:
            core_outdir_path = out_dir_path
        mirror_path = core_outdir_path / f"{vid}{ext}"
        mirror_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Build an explicit export dataframe that guarantees all flexis_* columns are kept ---
        base_cols = [
            "time_utc", "ts_ms", "timestamp",  # absolute time columns created above
            "t_ms", "t_s", "t_abs_s",          # relative time if present
            "lat", "lon", "speed_mps", "road_type", "is_service",
        ]

        # Ensure a fixed canonical Flexis schema is present (create empty defaults when missing)

        # 1bis) Enrichissement Flexis si colonnes manquantes ou vides (fallback non destructif)
        df = _ensure_flexis_columns(df, ctx)

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
        logger.debug("[DEBUG] Colonnes flexis_* avant export:")
        for col in required_flexis:
            if col in df.columns:
                logger.debug(f"  {col}: {df[col].head()}")

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

        # Preserve existing flexis_* from upstream stages; only fill blanks in df_out
        for _col in [c for c in df.columns if isinstance(c, str) and c.startswith("flexis_")]:
            try:
                if _col not in df_out.columns:
                    df_out[_col] = df[_col]
                else:
                    _s_out = df_out[_col].astype(object)
                    _s_src = df[_col].astype(object)
                    _mask_blank = _s_out.isna() | (_s_out.astype(str).str.strip() == "")
                    _s_out.loc[_mask_blank] = _s_src.loc[_mask_blank]
                    df_out[_col] = _s_out
            except Exception:
                pass

        # DEBUG: preview just before writing
        logger.debug("[DEBUG] Colonnes flexis_* juste avant écriture (post-merge):")
        for _col in [c for c in df_out.columns if isinstance(c, str) and c.startswith("flexis_")]:
            try:
                _s = pd.Series(df_out[_col])
                logger.debug(f"  {_col}: non_na={int(_s.notna().sum())}, unique={int(_s.nunique(dropna=True))}")
                logger.debug(_s.head().to_string())
            except Exception as __e:
                logger.debug(f"  {_col}: preview failed: {__e}")

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
            # Respect explicit extension even if it is not .csv (e.g. .csb as requested)
            csv_path = out_path
            df_out.to_csv(csv_path, index=False)
            wrote_paths.append(str(csv_path))
            # Mirror next to core HTML/Map reports, for immediate discoverability
            try:
                df_out.to_csv(mirror_path, index=False)
                wrote_paths.append(str(mirror_path))
            except Exception as _e:
                logger.debug(f"[FlexisExporter] Mirror write skipped: {type(_e).__name__}: {_e}")

        logger.info(f"[FlexisExporter] Wrote {', '.join(wrote_paths)}")
        # Force a visible stdout line (some runners filter logging levels)
        try:
            print("[FlexisExporter] Wrote:")
            for p in wrote_paths:
                print(" -", p)
        except Exception:
            pass

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

# Backward-compatibility alias
class FlexisExporter(Stage):
    pass
