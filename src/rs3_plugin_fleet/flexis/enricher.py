# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_datetime64tz_dtype
from zoneinfo import ZoneInfo

_DEF_PREFIX = "flexis_"

def _to_camel_from_snake(snake: str) -> str:
    return "".join(p.capitalize() for p in snake.split("_"))

def _aliases_for(target_snake: str) -> set[str]:
    if not target_snake.startswith(_DEF_PREFIX):
        target_snake = _DEF_PREFIX + target_snake
    camel = "Flexis" + _to_camel_from_snake(target_snake[len(_DEF_PREFIX):])
    compact = target_snake.replace("_", "")
    return {target_snake, camel, compact}

def _equiv_key(name: str) -> str:
    return name.replace("_", "").replace("-", "").replace(" ", "").lower()

def _find_existing_equiv(df: pd.DataFrame, target_snake: str) -> str | None:
    aliases = {_equiv_key(a) for a in _aliases_for(target_snake)}
    for c in df.columns:
        if _equiv_key(c) in aliases:
            return c
    return None

def _upsert_series(df: pd.DataFrame, target_snake: str, values) -> None:
    """
    Insert or replace a column, forcing alignment on df.index and using object dtype
    to ensure persistence of heterogeneous values across the pipeline.
    Also prints a small debug summary.
    """
    # Normalize target name with the "flexis_" prefix if missing
    if not isinstance(target_snake, str):
        raise TypeError("_upsert_series target_snake must be a string")
    if not target_snake.startswith(_DEF_PREFIX):
        target_snake = _DEF_PREFIX + target_snake

    # Ensure `values` is a pandas Series aligned on df.index and cast to object
    if not isinstance(values, pd.Series):
        values = pd.Series(values, index=df.index)
    values = values.reindex(df.index).astype(object)

    # Replace directly the target column
    df[target_snake] = values

    # Debug verification
    try:
        print(f"_upsert_series({target_snake}): non_na={df[target_snake].notna().sum()}, unique={df[target_snake].nunique(dropna=True)}")
    except Exception:
        pass

_ALIAS_TO_CANON = {
    # gardé uniquement pour la normalisation interne ; on exporte seulement flexis_*
    "FlexisRoadType": "flexis_road_type",
    "FlexisRoadCurve": "flexis_road_curve_radius_m",
    "FlexisNavigationEvents": "flexis_nav_event",
    "FlexisRoadInfrastructure": "flexis_infra_event",
    "FlexisNightCondition": "flexis_night",
    "FlexisDrivingBehaviour": "flexis_driver_event",
    "FlexisTraffic": "flexis_traffic_level",
    "FlexisWeatherCondition": "flexis_weather",
    "FlexisDeliveryStatus": "flexis_delivery_status",
    "FlexisPopulationDensity": "flexis_population_density_km2",
}
_ALIAS_NORM = {_equiv_key(k): v for k, v in _ALIAS_TO_CANON.items()}

def _canonicalize_flexis_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _blank_mask(s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series([True] * len(df), index=df.index)
        # consider NaN or empty-string-as-text as blank
        if s.dtype == 'O':
            # consider NaN, empty and whitespace-only strings as blank
            return s.isna() | (s.astype(str).str.strip() == "")
        try:
            return s.isna()
        except Exception:
            return pd.Series([False] * len(df), index=df.index)

    for c in list(df.columns):
        key = _equiv_key(c)
        if key in _ALIAS_NORM:
            target = _ALIAS_NORM[key]
            src = df[c]
            if target in df.columns:
                # merge: keep existing canonical values; only fill blanks from alias
                tgt = df[target]
                mask_fill = _blank_mask(tgt) & (~_blank_mask(src))
                if mask_fill.any():
                    tmp = tgt.copy()
                    tmp.loc[mask_fill] = src.loc[mask_fill]
                    _upsert_series(df, target, tmp)
            else:
                # no canonical yet ⇒ adopt alias as canonical
                _upsert_series(df, target, src)
            # drop the alias column to avoid later overwrites
            if c in df.columns and c != target:
                try:
                    df.drop(columns=[c], inplace=True)
                except Exception:
                    pass
    return df

def _drop_non_canonical_flexis(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for c in list(df.columns):
        lc = c.lower()
        if lc.startswith("flexis") and not lc.startswith("flexis_"):
            to_drop.append(c)
        elif lc.startswith("flexi") and not lc.startswith("flexis_"):
            to_drop.append(c)
    if to_drop:
        try:
            df.drop(columns=to_drop, inplace=True)
        except Exception:
            pass
    return df

def _ensure_non_empty_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Force des valeurs par défaut robustes pour les colonnes flexis_*, afin d’éviter des colonnes vides/NaN à l’export."""
    def _fill_str(col: str, default: str):
        if col in df.columns:
            s = df[col].astype(object)
            mask_empty = s.isna() | (s.astype(str).str.strip() == "")
            if mask_empty.any():
                s.loc[mask_empty] = default
            df[col] = s

    def _fill_num(col: str, default: float):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    _fill_str("flexis_road_type", "unknown")
    _fill_str("flexis_nav_event", "")
    _fill_str("flexis_infra_event", "")
    _fill_str("flexis_driver_event", "")
    _fill_str("flexis_traffic_level", "unknown")
    _fill_str("flexis_weather", "unknown")
    _fill_str("flexis_delivery_status", "en_route")

    _fill_num("flexis_population_density_km2", 600.0)

    # Rayon de courbure: lissage simple pour éviter des NaN initiaux/finaux
    if "flexis_road_curve_radius_m" in df.columns:
        rad = pd.to_numeric(df["flexis_road_curve_radius_m"], errors="coerce")
        df["flexis_road_curve_radius_m"] = rad.bfill().ffill()

    return df

# --------- Finalize flexis defaults: hard fill and strip ----------
def _finalize_flexis_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dernier filet de sécurité AVANT return :
    - supprime les espaces,
    - remplace les vides/NaN par des valeurs par défaut sémantiques,
    - assure un rayon de courbure non-NaN,
    - recalcule la densité si besoin,
    - normalise la colonne booléenne flexis_night.
    """
    df = df.copy()

    def _hard_fill_str(col: str, default: str):
        if col in df.columns:
            s = df[col].astype(object)
            # Convert to str, strip, and remove textual NaNs
            s = s.where(~s.isna(), "")
            s = s.astype(str).str.strip()
            s = s.replace({"nan": "", "None": ""})
            s = s.replace("", default)
            df[col] = s

    def _hard_fill_num(col: str, default: float):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.bfill().ffill()
            s = s.fillna(default)
            df[col] = s

    def _hard_fill_bool(col: str, default: bool):
        if col in df.columns:
            # Handle common representations (0/1, 'true'/'false', NaN)
            s = df[col]
            try:
                # First coerce to numeric to capture 0/1, then fill and cast to bool
                s_num = pd.to_numeric(s, errors="coerce")
                s_num = s_num.fillna(int(default)).astype(int)
                df[col] = s_num.astype(bool)
            except Exception:
                # Fallback: map string representations
                s_str = s.astype(str).str.strip().str.lower()
                s_bool = s_str.map({"true": True, "1": True, "yes": True, "y": True,
                                    "false": False, "0": False, "no": False, "n": False})
                s_bool = s_bool.fillna(default)
                df[col] = s_bool.astype(bool)

    # libellés
    _hard_fill_str("flexis_road_type", "unknown")
    _hard_fill_str("flexis_nav_event", "")
    _hard_fill_str("flexis_infra_event", "")
    _hard_fill_str("flexis_driver_event", "")
    _hard_fill_str("flexis_traffic_level", "unknown")
    _hard_fill_str("flexis_weather", "unknown")
    _hard_fill_str("flexis_delivery_status", "en_route")

    # booléen
    _hard_fill_bool("flexis_night", False)

    # numériques
    _hard_fill_num("flexis_road_curve_radius_m", 5000.0)

    # densité : si NaN, réévaluer via l'heuristique en utilisant road_type
    if "flexis_population_density_km2" in df.columns:
        dens = pd.to_numeric(df["flexis_population_density_km2"], errors="coerce")
        if dens.isna().any():
            base_rt = df.get("flexis_road_type", pd.Series(["unknown"] * len(df), index=df.index))
            dens2 = _osm_heuristic_density(base_rt)
            df["flexis_population_density_km2"] = dens.fillna(dens2).fillna(600.0)
    else:
        # Si la colonne n'existe pas, la créer à partir de l'heuristique
        base_rt = df.get("flexis_road_type", pd.Series(["unknown"] * len(df), index=df.index))
        df["flexis_population_density_km2"] = _osm_heuristic_density(base_rt).fillna(600.0)

    return df

# --------- helper: pick first available column from candidates --------------
def _pick_cols(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --------- helper: pick best non-empty column among candidates --------------
def _pick_best_non_empty(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Parmi plusieurs colonnes candidates, choisit celle avec le plus de valeurs non vides.
    Considère NaN, 'nan', 'None' et chaînes vides/espaces comme vides."""
    best_col = None
    best_score = -1
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(object)
            empty = s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None"]))
            score = int((~empty).sum())
            if score > best_score:
                best_score = score
                best_col = c
    return best_col

# --------- extraction profils trafic/météo depuis ctx (core2) ----------------
def _ctx_first(ctx: Any, *paths, default=None):
    def _get_one(root, path):
        cur = root
        for part in str(path).split('.'):
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
        return cur
    for p in paths:
        try:
            val = _get_one(ctx, p)
            if val not in (None, "", [], {}):
                return val
        except Exception:
            continue
    return default

def _extract_schedules_from_ctx(ctx: Any) -> tuple[Any, Any]:
    """Retourne (traffic_profile, weather_timeline) depuis ctx et son YAML."""
    if ctx is None:
        return None, None

    tp = _ctx_first(ctx, "traffic_profile", "config.traffic_profile", "cfg.traffic_profile", default=None)
    wt = _ctx_first(ctx, "weather_timeline", "config.weather_timeline", "cfg.weather_timeline", default=None)

    cfg = getattr(ctx, "config", None)
    if not isinstance(cfg, dict):
        cfg = getattr(ctx, "cfg", None) if isinstance(getattr(ctx, "cfg", None), dict) else getattr(ctx, "cfg", {})

    if isinstance(cfg, dict):
        profiles = cfg.get("profiles") or {}
        active_profile = _ctx_first(ctx, "profile", "profile_name", default=None)
        if not active_profile:
            vehicles = cfg.get("vehicles") or []
            if isinstance(vehicles, list) and vehicles:
                v0 = vehicles[0] or {}
                active_profile = v0.get("profile")
        if not active_profile and isinstance(profiles, dict) and len(profiles) == 1:
            active_profile = next(iter(profiles.keys()))
        if active_profile and isinstance(profiles, dict):
            prof_cfg = profiles.get(active_profile) or {}
            tp = tp or prof_cfg.get("traffic_profile")
            wt = wt or prof_cfg.get("weather_timeline")

    # Fallback: if still missing, scan all profiles for first match
    if (not tp or tp == []) or (not wt or wt == []):
        if isinstance(cfg, dict):
            profiles = cfg.get("profiles") or {}
            if isinstance(profiles, dict):
                for name, prof in profiles.items():
                    if isinstance(prof, dict):
                        cand_tp = prof.get("traffic_profile")
                        cand_wt = prof.get("weather_timeline")
                        if (not tp or tp == []) and cand_tp:
                            tp = cand_tp
                        if (not wt or wt == []) and cand_wt:
                            wt = cand_wt
                        if tp and wt:
                            break

    return tp, wt

# ---------- helpers temps / events ------------------------------------------
def _parse_hhmm(s: str) -> int:
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0

def _label_from_schedule(mins: pd.Series, schedule: Any, key: str) -> pd.Series:
    try:
        idx = mins.index
    except AttributeError:
        idx = pd.RangeIndex(len(mins))
    if schedule is None:
        return pd.Series("", index=idx)

    if isinstance(schedule, (tuple, set)):
        schedule = list(schedule)
    elif isinstance(schedule, dict):
        schedule = [schedule]
    elif hasattr(schedule, "to_dict") and not isinstance(schedule, list):
        try:
            schedule = schedule.to_dict(orient="records")  # type: ignore
        except Exception:
            schedule = [schedule]
    if not isinstance(schedule, list):
        schedule = [schedule]
    if len(schedule) == 0:
        return pd.Series("", index=idx)

    lab = np.array([""] * len(idx), dtype=object)
    for item in schedule:
        try:
            f = _parse_hhmm(str(item.get("from", "00:00")))
            t = _parse_hhmm(str(item.get("to", "00:00")))
            val = item.get(key) or item.get("level") or item.get("weather") or ""
        except AttributeError:
            continue
        mask = (mins >= f) & (mins < t) if t >= f else ((mins >= f) | (mins < t))
        lab[getattr(mask, "values", mask)] = val
    return pd.Series(lab, index=idx)

def _is_night_from_minutes(mins: pd.Series, night_bounds=(22 * 60, 6 * 60)) -> pd.Series:
    start, end = night_bounds
    mask = (mins >= start) & (mins < end) if start <= end else ((mins >= start) | (mins < end))
    return pd.Series(getattr(mask, "values", mask), index=mins.index)

def _bearing(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.cos(lat2) + np.sin(lat1) * np.sin(lat2) * np.cos(lon2 - lon1)
    th = np.degrees(np.arctan2(y, x))
    return (th + 360.0) % 360.0

def _delta_heading(h):
    d = np.diff(h, prepend=h[0])
    d = (d + 180.0) % 360.0 - 180.0
    return d

def _radius_from_geometry(lat, lon):
    if len(lat) < 3:
        return np.full_like(lat, np.nan, dtype=float)
    brg = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dhead = _delta_heading(brg)
    R = 6371000.0
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = np.radians(lat[1:] - lat[:-1])
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    step = 2 * R * np.arcsin(np.sqrt(a))
    rad = np.full_like(lat, np.nan, dtype=float)
    denom = np.radians(np.abs(dhead)) + 1e-9
    rseg = np.clip(step / denom, 0, 5000.0)
    rad[1:] = rseg
    if len(rad) > 2:
        rad[0] = rad[1]
        rad[-1] = rad[-2]
    return rad

def _osm_heuristic_density(road_type: pd.Series) -> pd.Series:
    table = {
        "residential": 3000,
        "tertiary": 2000,
        "secondary": 1500,
        "primary": 800,
        "service": 1200,
        "trunk": 400,
        "motorway": 200,
        "unknown": 600,
    }
    if isinstance(road_type, pd.Series):
        s = road_type.astype(str).str.strip().replace("", "unknown")
        mapped = s.map(table)
        return mapped.fillna(600).astype(float)
    idx = getattr(road_type, "index", None)
    return pd.Series(600.0, index=idx, dtype=float)

# ---------- Enricher ---------------------------------------------------------
@dataclass
class FlexisConfig:
    traffic_profile: Any
    weather_timeline: Any
    infra_prob_per_km: Dict[str, float]
    driver_rate_per_h: Dict[str, float]
    population_density_mode: str = "osm_heuristic"
    timezone: str = "Europe/Paris"
    add_legacy_aliases: bool = False
    export_after_enrich: bool = False
    export_outdir: str = "data/simulations/default"
    export_filename: str = "flexis_final.csv"

class FlexisEnricher:
    name = "flexis"
    version = "0.6.0"

    def __init__(self, config: Dict[str, Any]):
        self.cfg = FlexisConfig(
            traffic_profile=config.get("traffic_profile", []),
            weather_timeline=config.get("weather_timeline", []),
            infra_prob_per_km=config.get("infra_probability_per_km", {}),
            driver_rate_per_h=config.get("driver_events_rate_per_hour", {}),
            population_density_mode=config.get("population_density_mode", "osm_heuristic"),
            timezone=config.get("timezone", "Europe/Paris"),
            add_legacy_aliases=False,
            export_after_enrich=config.get("export_after_enrich", False),
            export_outdir=config.get("export_outdir", "data/simulations/default"),
            export_filename=config.get("export_filename", "flexis_final.csv"),
        )

    def _resolve_time(self, df: pd.DataFrame, ctx: Any | None) -> pd.DatetimeIndex:
        """Produit un DatetimeIndex UTC tz-aware en détectant timestamp/époques/relatifs."""
        tz = ZoneInfo(self.cfg.timezone)

        # 1) timestamp si présent
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            if is_datetime64tz_dtype(ts):
                dt = pd.to_datetime(ts, errors="coerce").tz_convert("UTC")
                return pd.DatetimeIndex(dt)
            if is_datetime64_any_dtype(ts):
                dt = pd.to_datetime(ts, errors="coerce")
                dt = dt.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
                return pd.DatetimeIndex(dt)
            if is_numeric_dtype(ts):
                s_num = pd.to_numeric(ts, errors="coerce")
                m = float(s_num.max()) if s_num.notna().any() else -1
                if m >= 1e12:
                    return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
                if m >= 1e9:
                    return pd.to_datetime(s_num, unit="s", utc=True, errors="coerce")
        # 2) époques dédiées
        for col, unit, thr in (("ts_ms", "ms", 1e11), ("ts_s", "s", 1e9)):
            if col in df.columns and is_numeric_dtype(df[col]):
                s = pd.to_numeric(df[col], errors="coerce")
                q = s.quantile(0.9)
                if pd.notna(q) and q >= thr:
                    return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")

        # 3) temps relatifs (secondes)
        rel_s = None
        if "t_ms" in df.columns and is_numeric_dtype(df["t_ms"]):
            rel_s = pd.to_numeric(df["t_ms"], errors="coerce") / 1000.0
        elif "t_abs_s" in df.columns and is_numeric_dtype(df["t_abs_s"]):
            rel = pd.to_numeric(df["t_abs_s"], errors="coerce")
            q = rel.quantile(0.9)
            rel_s = rel if not (pd.notna(q) and q >= 1e9) else None
            if rel_s is None:
                return pd.to_datetime(rel, unit="s", utc=True, errors="coerce")
        elif "t_s" in df.columns and is_numeric_dtype(df["t_s"]):
            rel_s = pd.to_numeric(df["t_s"], errors="coerce")

        base = None
        if ctx is not None:
            for attr in ("sim_start", "start_time", "t0", "base_time"):
                v = getattr(ctx, attr, None)
                if v is not None:
                    try:
                        base = pd.Timestamp(v)
                        break
                    except Exception:
                        pass
        if base is None:
            base = pd.Timestamp("2024-01-01 08:00", tz=tz)
        if base.tzinfo is None:
            base = base.tz_localize(tz)

        if rel_s is not None:
            dt_local = base + pd.to_timedelta(rel_s, unit="s")
            return pd.to_datetime(dt_local).tz_convert("UTC")

        # 4) dernier recours: cadence
        n = len(df)
        hz = None
        if ctx is not None:
            for attr in ("hz", "cadence_hz"):
                v = getattr(ctx, attr, None)
                if isinstance(v, (int, float)) and v > 0:
                    hz = float(v)
                    break
        if hz is None:
            hz = 10.0
        dt_local = base + pd.to_timedelta(np.arange(n) / hz, unit="s")
        return pd.to_datetime(dt_local).tz_convert("UTC")

    def apply(self, df: pd.DataFrame, context: Any | None = None) -> pd.DataFrame:
        df = df.copy()
        ctx = context
        try:
            print("FlexisEnricher: input columns:", list(df.columns))
            print("FlexisEnricher: available road type columns:", [c for c in df.columns if ("road" in c.lower()) or ("highway" in c.lower()) or ("osm" in c.lower())])
        except Exception:
            pass

        # altitude
        altitude = None
        for c in ("altitude", "elev_m", "elevation"):
            if c in df.columns:
                altitude = pd.to_numeric(df[c], errors="coerce")
                break
        df["altitude"] = altitude if altitude is not None else np.nan

        # -- latitude/longitude (prendre le meilleur disponible)
        lat_col = _pick_cols(df, ["lat", "latitude", "y"])
        lon_col = _pick_cols(df, ["lon", "lng", "longitude", "x"])

        # road_type -> flexis_road_type (vides => "unknown")
        # Choisit la meilleure colonne disponible (max contenu non vide)
        road_candidates = ["road_type", "osm_highway", "highway", "osm_road_type"]
        src_rt = _pick_best_non_empty(df, road_candidates)
        try:
            print("FlexisEnricher: road_type source:", src_rt)
            print("FlexisEnricher: traffic_profile (cfg):", self.cfg.traffic_profile)
            print("FlexisEnricher: weather_timeline (cfg):", self.cfg.weather_timeline)
        except Exception:
            pass
        if src_rt:
            rt = df[src_rt].astype(str)
            # normalisation et nettoyage
            rt = rt.replace({"None": np.nan, "nan": np.nan}).fillna("")
            rt = rt.str.strip().replace("", "unknown")
            _upsert_series(df, "flexis_road_type", rt)
        else:
            _upsert_series(df, "flexis_road_type", pd.Series(["unknown"] * len(df), index=df.index))

        # courbure (rayon)
        if lat_col and lon_col:
            radius = pd.Series(
                _radius_from_geometry(df[lat_col].values, df[lon_col].values),
                index=df.index,
            ).replace([np.inf, -np.inf], np.nan).bfill().ffill().clip(0, 5000)
        else:
            radius = pd.Series(np.nan, index=df.index)
        _upsert_series(df, "flexis_road_curve_radius_m", radius)

        # nav events
        nav = self._nav_events(df)
        _upsert_series(df, "flexis_nav_event", nav)

        # timeline absolu + minutes locales
        ts_abs = self._resolve_time(df, ctx)
        df["timestamp"] = ts_abs
        ts_local = ts_abs.tz_convert(ZoneInfo(self.cfg.timezone))
        mins = pd.Series((ts_local.hour * 60 + ts_local.minute).astype(int), index=df.index)

        # profils trafic / météo : préférence à la config passée au stage, sinon ctx.config/cfg.profile
        cfg_tp = self.cfg.traffic_profile
        cfg_wt = self.cfg.weather_timeline
        if (not cfg_tp) or (not cfg_wt):
            ctx_tp, ctx_wt = _extract_schedules_from_ctx(ctx)
            if not cfg_tp:
                cfg_tp = ctx_tp
            if not cfg_wt:
                cfg_wt = ctx_wt

        weather = _label_from_schedule(mins, cfg_wt, key="weather") if cfg_wt else pd.Series("", index=df.index)
        _upsert_series(df, "flexis_weather", weather)

        night = _is_night_from_minutes(mins, night_bounds=(22 * 60, 6 * 60))
        _upsert_series(df, "flexis_night", night)

        traffic = _label_from_schedule(mins, cfg_tp, key="level") if cfg_tp else pd.Series("", index=df.index)
        _upsert_series(df, "flexis_traffic_level", traffic)

        # autres features
        _upsert_series(df, "flexis_delivery_status", self._delivery_timeline(df, ctx))
        _upsert_series(df, "flexis_infra_event", self._infra_events(df))
        _upsert_series(df, "flexis_driver_event", self._driver_events(df))

        # densité population (inconnus => 600 par défaut)
        base_rt = df.get("flexis_road_type", df.get("road_type", pd.Series([""] * len(df), index=df.index)))
        base_rt = base_rt.astype(str).replace({"None": "", "nan": ""}).replace("", "unknown")
        pop = _osm_heuristic_density(base_rt).fillna(600)
        _upsert_series(df, "flexis_population_density_km2", pop)

        # (désactivé) canoniser & nettoyer pour éviter d'écraser les colonnes calculées
        # df = _canonicalize_flexis_columns(df)
        # df = _drop_non_canonical_flexis(df)

        # ENFORCE: valeurs par défaut souples
        df = _ensure_non_empty_defaults(df)
        # Renforcement final: nettoyage/valeurs par défaut strictes avant retour
        df = _finalize_flexis_defaults(df)

        # Filet de sécurité : force des valeurs par défaut si encore totalement vides
        for col in [
            "flexis_road_type",
            "flexis_traffic_level",
            "flexis_weather",
            "flexis_delivery_status",
        ]:
            if col in df.columns and pd.Series(df[col]).isna().all():
                df[col] = "unknown"
        if "flexis_population_density_km2" in df.columns and pd.Series(df["flexis_population_density_km2"]).isna().all():
            df["flexis_population_density_km2"] = 600.0

        # --- Debug résumé pour validation runtime ---
        try:
            def _summ(col):
                s = df.get(col)
                if s is None:
                    return {"present": False}
                s_series = pd.Series(s)
                return {
                    "present": True,
                    "non_na": int(s_series.notna().sum()),
                    "unique": int(s_series.nunique(dropna=True)),
                    "head": s_series.head(3).tolist(),
                    "value_counts_top": s_series.astype(str).value_counts(dropna=False).head(3).to_dict(),
                }
            summary_cols = [
                "flexis_road_type",
                "flexis_traffic_level",
                "flexis_weather",
                "flexis_population_density_km2",
                "flexis_road_curve_radius_m",
            ]
            print("FlexisEnricher summary:", {c: _summ(c) for c in summary_cols})
        except Exception:
            pass

        return df

    def process(self, df: pd.DataFrame, ctx: Any = None) -> pd.DataFrame:
        return self.apply(df, ctx)

    def _nav_events(self, df: pd.DataFrame) -> pd.Series:
        lat_col = _pick_cols(df, ["lat", "latitude", "y"])
        lon_col = _pick_cols(df, ["lon", "lng", "longitude", "x"])
        if lat_col and lon_col:
            h = _bearing(df[lat_col].shift(1), df[lon_col].shift(1), df[lat_col], df[lon_col])
            dh = pd.Series(_delta_heading(h.values), index=df.index).abs()
        else:
            dh = pd.Series(0.0, index=df.index)

        v_col = _pick_cols(df, ["speed_mps", "speed", "v", "v_mps"])
        v = pd.to_numeric(df[v_col], errors="coerce") if v_col else pd.Series(np.nan, index=df.index)

        events = np.array([""] * len(df), dtype=object)
        events[(dh > 25) & (v >= 0.3)] = "turn"
        events[(v < 0.1) | (v.isna())] = "wait"
        return pd.Series(events, index=df.index)

    def _delivery_timeline(self, df: pd.DataFrame, context: Any) -> pd.Series:
        if "is_service" in df.columns:
            sv = df["is_service"].astype(bool)
            ev = np.array(["en_route"] * len(df), dtype=object)
            ev[sv.values] = "delivery_in_progress"
            return pd.Series(ev, index=df.index)
        return pd.Series("en_route", index=df.index, dtype="object")

    def _infra_events(self, df: pd.DataFrame) -> pd.Series:
        if "acc_z" in df.columns:
            az = pd.to_numeric(df["acc_z"], errors="coerce").abs()
            th = max(az.quantile(0.98), 2.5) if az.notna().any() else 3.0
            tags = np.where(az > th, "bump", "")
            return pd.Series(tags, index=df.index)
        return pd.Series([""] * len(df), index=df.index)

    def _driver_events(self, df: pd.DataFrame) -> pd.Series:
        v = None
        for c in ("speed_mps", "speed", "v", "v_mps"):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                break
        if v is None or "timestamp" not in df.columns:
            return pd.Series([""] * len(df), index=df.index)
        ts_abs = self._resolve_time(df, None)
        t = ts_abs.view("int64") / 1e9  # seconds
        dv = v.diff()
        dt = pd.Series(t).diff().replace(0, np.nan)
        a = dv / dt
        tags = np.array([""] * len(df), dtype=object)
        tags[a < -2.5] = "harsh_brake"
        tags[a > 2.5] = "aggressive_accel"
        return pd.Series(tags, index=df.index)

    def run(self, ctx):
        df = getattr(ctx, "df", None)
        if df is None:
            df = getattr(ctx, "timeline", None)
        if df is None:
            return {"ok": False, "msg": "FlexisEnricher.run: no dataframe found in ctx"}
        try:
            out = self.process(df, ctx)
            # Renforcement final pour éviter toute régression en aval
            out = _finalize_flexis_defaults(out)
            # Force a deep copy to ensure persistence of computed flexis_* columns across stages
            out = out.copy(deep=True)

            # Debug: list and summarize flexis_* columns before attaching to context
            try:
                flexis_cols = [c for c in out.columns if str(c).startswith("flexis_")]
                print("FlexisEnricher.run: flexis_cols in out:", flexis_cols)
                for col in flexis_cols:
                    try:
                        non_na = int(pd.Series(out[col]).notna().sum())
                        unique = int(pd.Series(out[col]).nunique(dropna=True))
                    except Exception:
                        non_na = "?"
                        unique = "?"
                    print(f"  {col}: non_na={non_na}, unique={unique}")
            except Exception:
                pass

            # Optional: dedicated export of flexis_* right after enrichment to avoid later stage overwrites
            try:
                if getattr(self.cfg, "export_after_enrich", False):
                    flexis_cols = [c for c in out.columns if str(c).startswith("flexis_")]
                    if flexis_cols:
                        flexis_df = out[flexis_cols].copy()

                        # Coerce types and replace empties/NaN with robust defaults
                        for col in flexis_df.columns:
                            if flexis_df[col].dtype == "object":
                                s = flexis_df[col].astype(str)
                                s = s.replace("nan", "").replace("None", "").str.strip()
                                # Set semantic defaults
                                default = "unknown" if col not in ("flexis_nav_event", "flexis_infra_event", "flexis_driver_event") else ""
                                s = s.replace("", default)
                                flexis_df[col] = s
                            else:
                                # numeric columns
                                if col == "flexis_population_density_km2":
                                    flexis_df[col] = pd.to_numeric(flexis_df[col], errors="coerce").fillna(600.0)
                                elif col == "flexis_road_curve_radius_m":
                                    s = pd.to_numeric(flexis_df[col], errors="coerce")
                                    flexis_df[col] = s.bfill().ffill().fillna(5000.0)
                                else:
                                    flexis_df[col] = pd.to_numeric(flexis_df[col], errors="coerce").fillna(0)

                        outdir = getattr(self.cfg, "export_outdir", "data/simulations/default")
                        filename = getattr(self.cfg, "export_filename", "flexis_final.csv")
                        os.makedirs(outdir, exist_ok=True)
                        filepath = os.path.join(outdir, filename)
                        flexis_df.to_csv(filepath, index=False)

                        print(f"FlexisEnricher.run: exported {len(flexis_cols)} flexis_* columns to {filepath}")
                    else:
                        print("FlexisEnricher.run: no flexis_* columns to export")
            except Exception as _e:
                print(f"FlexisEnricher.run: export_after_enrich failed: {_e}")

            # Attach to context
            setattr(ctx, "df", out)
            setattr(ctx, "timeline", out)
            return {"ok": True, "msg": "flexis enrichment applied"}
        except Exception as e:
            return {"ok": False, "msg": f"flexis failed: {e}"}