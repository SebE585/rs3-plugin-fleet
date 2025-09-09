# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
)

# =============================================================================
# Helpers — normalisation de noms & alias (FlexisXXX → flexis_xxx), utilitaires
# =============================================================================

_DEF_PREFIX = "flexis_"

def _to_camel_from_snake(snake: str) -> str:
    parts = snake.split("_")
    return "".join(p.capitalize() for p in parts)

def _aliases_for(target_snake: str) -> set[str]:
    """
    Retourne l'ensemble des alias possibles pour une colonne flexis_foo_bar :
      - snake:  flexis_foo_bar
      - Camel:  FlexisFooBar
      - compact (sans underscore, tout en bas): flexisfoobar
    """
    if not target_snake.startswith(_DEF_PREFIX):
        target_snake = _DEF_PREFIX + target_snake
    camel = "Flexis" + _to_camel_from_snake(target_snake[len(_DEF_PREFIX):])
    compact = target_snake.replace("_", "")
    return {target_snake, camel, compact}

def _equiv_key(name: str) -> str:
    """Clé d’équivalence insensible au style (snake/camel/espaces/traits)."""
    return name.replace("_", "").replace("-", "").replace(" ", "").lower()

def _find_existing_equiv(df: pd.DataFrame, target_snake: str) -> str | None:
    aliases = { _equiv_key(a) for a in _aliases_for(target_snake) }
    for c in df.columns:
        if _equiv_key(c) in aliases:
            return c
    return None

def _upsert_series(df: pd.DataFrame, target_snake: str, values: pd.Series) -> None:
    """
    Crée/écrase la colonne canonique snake_case (flexis_...), et supprime l’alias
    existant (CamelCase/typo) s’il y en a un.
    """
    if not target_snake.startswith(_DEF_PREFIX):
        target_snake = _DEF_PREFIX + target_snake
    values = values.reindex(df.index)
    existing = _find_existing_equiv(df, target_snake)
    df[target_snake] = values
    if existing and existing != target_snake and existing in df.columns:
        try:
            df.drop(columns=[existing], inplace=True)
        except Exception:
            pass

# alias historiques/typos → canon
_ALIAS_TO_CANON = {
    "FlexisRoadType": "flexis_road_type",
    "FlexisRoadCurve": "flexis_road_curve_radius_m",
    "FlexisNavigationEvents": "flexis_nav_event",
    "FlexisRoadInfrastrcture": "flexis_infra_event",          # typo
    "FlexisNightCondition": "flexis_night",
    "FlexisDrivingBeheviour": "flexis_driver_event",          # typo
    "FlexiTraffic": "flexis_traffic_level",                   # manque 's'
    "FlexiWetaherCondition": "flexis_weather",                # typo
    "Fexis Delivery Status": "flexis_delivery_status",        # typos/espaces
    "FlexiSpulationDensity": "flexis_population_density_km2", # typo
}
_ALIAS_NORM = { _equiv_key(k): v for k, v in _ALIAS_TO_CANON.items() }

def _canonicalize_flexis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les colonnes alias vers la forme canonique flexis_* et supprime les doublons."""
    for c in list(df.columns):
        key = _equiv_key(c)
        if key in _ALIAS_NORM:
            target = _ALIAS_NORM[key]
            _upsert_series(df, target, df[c])
            if c in df.columns and c != target:
                try:
                    df.drop(columns=[c], inplace=True)
                except Exception:
                    pass
    return df

def _drop_non_canonical_flexis(df: pd.DataFrame) -> pd.DataFrame:
    """Balayage final : toute colonne Flexi*/Flexis* non 'flexis_' est supprimée."""
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

# =============================================================================
# Helpers — temps, navigation, IMU, courbure, densité population
# =============================================================================

def _parse_hhmm(s: str) -> int:
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0

def _tod_minutes(ts: pd.Series) -> pd.Series:
    """
    Minutes depuis minuit pour chaque timestamp.
    Accepte : epoch seconds/millis (numérique), datetimes pandas (avec/sans tz), ou strings parsables.
    """
    try:
        if is_numeric_dtype(ts):
            mins = ((pd.to_numeric(ts, errors="coerce") / 60.0) % (24 * 60)).astype("Int64").fillna(0).astype(int)
            return mins
        # datetimes (tz-aware ou non) OU strings à parser
        if is_datetime64_any_dtype(ts) or is_datetime64tz_dtype(ts):
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
        return (dt.dt.hour * 60 + dt.dt.minute).astype("Int64").fillna(0).astype(int)
    except Exception:
        return pd.Series(0, index=ts.index)

def _label_from_schedule(mins: pd.Series, schedule, key: str) -> pd.Series:
    """
    Map minutes-of-day → labels selon une liste de fenêtres [{from,to,level|weather}], avec wrap minuit géré.
    `schedule` peut être list/tuple/set/None/objet — on le normalise en liste.
    """
    # Normalisation en liste
    if schedule is None:
        return pd.Series("", index=mins.index)
    if isinstance(schedule, (tuple, set)):
        schedule = list(schedule)
    elif not isinstance(schedule, list):
        # dict/DataFrame/objet → liste singleton
        schedule = [schedule]
    if len(schedule) == 0:
        return pd.Series("", index=mins.index)

    lab = np.array([""] * len(mins), dtype=object)
    for item in schedule:
        try:
            f = _parse_hhmm(item.get("from", "00:00"))
            t = _parse_hhmm(item.get("to", "00:00"))
            val = item.get(key) or item.get("level") or item.get("weather") or ""
        except AttributeError:
            # au cas où item n'est pas un dict
            f, t, val = 0, 0, ""
        mask = (mins >= f) & (mins < t) if t >= f else ((mins >= f) | (mins < t))
        lab[getattr(mask, "values", mask)] = val
    return pd.Series(lab, index=mins.index)

def _is_night_from_minutes(mins: pd.Series, night_bounds=(20*60, 6*60)) -> pd.Series:
    start, end = night_bounds
    mask = (mins >= start) & (mins < end) if start <= end else ((mins >= start) | (mins < end))
    return pd.Series(mask.values, index=mins.index)

def _bearing(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.cos(lat2) + np.sin(lat1) * np.sin(lat2) * np.cos(lon2 - lon1)
    th = np.degrees(np.arctan2(y, x))
    return (th + 360.0) % 360.0

def _delta_heading(h):
    d = np.diff(h, prepend=h[0])
    d = (d + 180.0) % 360.0 - 180.0
    return d

def _radius_from_geometry(lat, lon):
    """Rayon de courbure approximé (m) à partir des caps successifs."""
    if len(lat) < 3:
        return np.full_like(lat, np.nan, dtype=float)
    brg = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dhead = _delta_heading(brg)
    R = 6371000.0
    lat1 = np.radians(lat[:-1]); lat2 = np.radians(lat[1:])
    dlat = np.radians(lat[1:] - lat[:-1]); dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    step = 2 * R * np.arcsin(np.sqrt(a))
    rad = np.full_like(lat, np.nan, dtype=float)
    denom = np.radians(np.abs(dhead)) + 1e-9
    rseg = np.clip(step / denom, 0, 5000.0)
    rad[1:] = rseg
    if len(rad) > 2:
        rad[0] = rad[1]; rad[-1] = rad[-2]
    return rad

def _osm_heuristic_density(road_type: pd.Series) -> pd.Series:
    """Heuristique simple densité (hab/km²) selon le type de voie."""
    table = {
        "residential": 3000,
        "tertiary": 2000,
        "secondary": 1500,
        "primary": 800,
        "service": 1200,
        "trunk": 400,
        "motorway": 200,
    }
    if isinstance(road_type, pd.Series):
        s = road_type.astype(str)
    else:
        try:
            idx = road_type.index  # type: ignore[attr-defined]
        except Exception:
            idx = None
        s = pd.Series("", index=idx)
    return s.map(table).fillna(600)

# =============================================================================
# Enricher — Config & logique
# =============================================================================

@dataclass
class FlexisConfig:
    traffic_profile: list
    weather_timeline: list
    infra_prob_per_km: Dict[str, float]
    driver_rate_per_h: Dict[str, float]
    population_density_mode: str = "osm_heuristic"

class FlexisEnricher:
    """
    Source de vérité unique pour les colonnes flexis_*.

    Colonnes produites (snake_case uniquement) :
      - altitude
      - flexis_road_type
      - flexis_road_curve_radius_m
      - flexis_nav_event
      - flexis_weather
      - flexis_night
      - flexis_traffic_level
      - flexis_delivery_status
      - flexis_infra_event
      - flexis_driver_event
      - flexis_population_density_km2
    """
    name = "flexis"
    version = "0.3.0"

    def __init__(self, config: Dict[str, Any]):
        self.cfg = FlexisConfig(
            traffic_profile=config.get("traffic_profile", []),
            weather_timeline=config.get("weather_timeline", []),
            infra_prob_per_km=config.get("infra_probability_per_km", {}),
            driver_rate_per_h=config.get("driver_events_rate_per_hour", {}),
            population_density_mode=config.get("population_density_mode", "osm_heuristic"),
        )

    # -------------------- API principale --------------------------------------
    def apply(self, df: pd.DataFrame, context: Dict[str, Any] | None = None) -> pd.DataFrame:
        df = df.copy()
        context = context or {}

        # 0) altitude (aliases courants)
        altitude = None
        for c in ("altitude", "elev_m", "elevation"):
            if c in df.columns:
                altitude = pd.to_numeric(df[c], errors="coerce")
                break
        if (altitude is not None) and ("altitude" not in df.columns):
            df["altitude"] = altitude
        elif altitude is None:
            df["altitude"] = np.nan

        # 1) type de voie (si présent)
        if "road_type" in df.columns:
            mapped = df["road_type"].fillna("unknown").astype(str)
            _upsert_series(df, "flexis_road_type", mapped)

        # 2) courbure (m)
        if {"lat", "lon"}.issubset(df.columns):
            radius = pd.Series(
                _radius_from_geometry(df["lat"].to_numpy(), df["lon"].to_numpy()),
                index=df.index,
            ).replace([np.inf, -np.inf], np.nan).bfill().ffill().clip(0, 5000)
        else:
            radius = pd.Series(np.nan, index=df.index)
        _upsert_series(df, "flexis_road_curve_radius_m", radius)

        # 3) événements navigation (turn/wait)
        nav = self._nav_events(df)
        _upsert_series(df, "flexis_nav_event", nav)

        # 4) weather + night selon timelines si timestamp existe
        if "timestamp" in df.columns:
            mins = _tod_minutes(df["timestamp"])  # robuste : numeric/datetime/str
        else:
            mins = pd.Series(0, index=df.index)

        wt = self.cfg.weather_timeline
        weather = _label_from_schedule(mins, wt, key="weather") if (wt is not None and len(wt) > 0) else pd.Series("", index=df.index)
        _upsert_series(df, "flexis_weather", weather)

        night = _is_night_from_minutes(mins)
        _upsert_series(df, "flexis_night", night)

        # 5) trafic
        tp = self.cfg.traffic_profile
        traffic = _label_from_schedule(mins, tp, key="level") if (tp is not None and len(tp) > 0) else pd.Series("", index=df.index)
        _upsert_series(df, "flexis_traffic_level", traffic)

        # 6) statut livraison
        delivery = self._delivery_timeline(df, context)
        _upsert_series(df, "flexis_delivery_status", delivery)

        # 7) infra & driver
        infra = self._infra_events(df)
        _upsert_series(df, "flexis_infra_event", infra)

        driver = self._driver_events(df)
        _upsert_series(df, "flexis_driver_event", driver)

        # 8) densité population
        if (self.cfg.population_density_mode == "osm_heuristic") or (self.cfg.population_density_mode is None):
            base_rt = df["flexis_road_type"] if "flexis_road_type" in df.columns else pd.Series([""] * len(df), index=df.index)
            pop = _osm_heuristic_density(base_rt)
        else:
            pop = pd.Series("", index=df.index)  # hook pour modes raster ultérieurs
        _upsert_series(df, "flexis_population_density_km2", pop)

        # 9) nettoyage alias → snake_case unique
        df = _canonicalize_flexis_columns(df)
        df = _drop_non_canonical_flexis(df)

        return df

    # -------------------- Helpers de features ---------------------------------
    def _nav_events(self, df: pd.DataFrame) -> pd.Series:
        """Heuristique légère : 'turn' si |Δcap|>25° et vitesse >= 0.3 m/s ; 'wait' si vitesse ~ 0."""
        if {"lat", "lon"}.issubset(df.columns):
            h = _bearing(df["lat"].shift(1), df["lon"].shift(1), df["lat"], df["lon"])
            dh = pd.Series(_delta_heading(h.to_numpy()), index=df.index).abs()
        else:
            dh = pd.Series(0.0, index=df.index)

        v = None
        for c in ("speed_mps", "speed", "v"):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce"); break
        if v is None:
            v = pd.Series(np.nan, index=df.index)

        events = np.array([""] * len(df), dtype=object)
        events[(dh > 25) & (v >= 0.3)] = "turn"
        events[(v < 0.1) | (v.isna())] = "wait"
        return pd.Series(events, index=df.index)

    def _delivery_timeline(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.Series:
        """Si df['is_service'] existe → 'delivery_in_progress' pendant le service, sinon 'en_route'."""
        if "is_service" in df.columns:
            sv = df["is_service"].astype(bool)
            ev = np.array(["en_route"] * len(df), dtype=object)
            ev[sv.values] = "delivery_in_progress"
            return pd.Series(ev, index=df.index)
        return pd.Series("en_route", index=df.index, dtype="object")

    def _infra_events(self, df: pd.DataFrame) -> pd.Series:
        """Détection simple 'bump' si acc_z présente et forte variabilité."""
        if "acc_z" in df.columns:
            az = pd.to_numeric(df["acc_z"], errors="coerce").abs()
            th = max(az.quantile(0.98), 2.5) if az.notna().any() else 3.0
            tags = np.where(az > th, "bump", "")
            return pd.Series(tags, index=df.index)
        return pd.Series([""] * len(df), index=df.index)

    def _driver_events(self, df: pd.DataFrame) -> pd.Series:
        """Détecte harsh_brake / aggressive_accel via dv/dt si speed + timestamp existent."""
        v = None
        for c in ("speed_mps", "speed", "v"):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce"); break
        if v is None or "timestamp" not in df.columns:
            return pd.Series([""] * len(df), index=df.index)

        # timestamp : accepte numérique/datetime/string (on convertit en seconds)
        ts = df["timestamp"]
        if is_numeric_dtype(ts):
            t = pd.to_numeric(ts, errors="coerce")
        else:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
            t = dt.astype("int64") / 1e9  # seconds float (ns → s)

        dv = v.diff()
        dt = pd.Series(t).diff().replace(0, np.nan)
        a = dv / dt
        tags = np.array([""] * len(df), dtype=object)
        tags[a < -2.5] = "harsh_brake"
        tags[a >  2.5] = "aggressive_accel"
        return pd.Series(tags, index=df.index)