# rs3_plugin_fleet/flexis/enricher.py
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from typing import Any

import numpy as np
import pandas as pd


# Rayon « très grand » utilisé pour section quasi-ligne droite
STRAIGHT_RADIUS_FALLBACK = 5000.0  # mètres
# Seuil « virage » : rayon < threshold -> virage
CURVE_RADIUS_THRESHOLD_M = 200.0
# Lissage pour cap/heading (en points)
HEADING_SMOOTH_N = 3
# Fenêtre rolling pour trafic (en secondes)
TRAFFIC_ROLLING_S = 30.0


# --- Time axis helpers ---
def _median_dt_seconds_from_series(ts: pd.Series) -> float:
    """Return median positive delta-t in seconds from a datetime-like series."""
    if ts is None or ts.empty:
        return 0.0
    try:
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        idx = pd.DatetimeIndex(t)
        if len(idx) < 2:
            return 0.0
        ns = idx.asi8
        dt = np.diff((ns - ns[0]) / 1e9, prepend=0.0)
        pos = dt > 0
        if not np.any(pos):
            return 0.0
        return float(np.median(dt[pos]))
    except Exception:
        return 0.0


def _infer_hz(df: pd.DataFrame) -> float | None:
    """
    Best-effort Hz inference from time columns (time_utc, timestamp, t_abs_s).
    Returns None if it cannot be inferred.
    """
    # Prefer explicit millisecond ticks if present
    if "ts_ms" in df.columns:
        try:
            ts = pd.to_numeric(df["ts_ms"], errors="coerce").astype("float64")
            d = ts.diff().dropna()
            d = d[d > 0]
            if len(d):
                med_ms = float(d.median())
                if med_ms > 0:
                    return 1000.0 / med_ms
        except Exception:
            pass

    # Try time_utc then timestamp
    for col in ("time_utc", "timestamp"):
        if col in df.columns:
            dt = _median_dt_seconds_from_series(df[col])
            if dt > 0:
                return 1.0 / dt

    # Fallback to t_abs_s
    if "t_abs_s" in df.columns:
        try:
            t = pd.to_numeric(df["t_abs_s"], errors="coerce").astype("float64")
            d = t.diff().dropna()
            d = d[d > 0]
            if len(d):
                med_s = float(d.median())
                if med_s > 0:
                    return 1.0 / med_s
        except Exception:
            pass

    return None


def _log_time_axis_health(logger: logging.Logger, df: pd.DataFrame, hz: float | None) -> None:
    """Log basic diagnostics about the time axis without mutating the DataFrame."""
    try:
        dup_ts_ms = int(df["ts_ms"].duplicated().sum()) if "ts_ms" in df.columns else -1
    except Exception:
        dup_ts_ms = -1

    mono_utc = None
    if "time_utc" in df.columns:
        try:
            t = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
            mono_utc = bool(t.is_monotonic_increasing)
            med_dt_s = _median_dt_seconds_from_series(df["time_utc"])
        except Exception:
            mono_utc = None
            med_dt_s = 0.0
    else:
        med_dt_s = 0.0

    logger.debug(
        "[FlexisEnricher] time axis: hz_inferred=%s, dt_med_s=%.3f, time_utc_monotone=%s, ts_ms_dups=%s",
        (f"{hz:.3f}" if hz else "None"),
        float(med_dt_s),
        mono_utc if mono_utc is not None else "NA",
        dup_ts_ms if dup_ts_ms >= 0 else "NA",
    )


@dataclass
class FlexisConfig:
    traffic_profile: Optional[List[Dict[str, str]]] = None
    weather_timeline: Optional[List[Dict[str, str]]] = None
    # colonnes possibles pour déterminer le type de route
    road_type_priority: Sequence[str] = ("road_type", "osm_highway")
    # libellés
    labels: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {
            "delivery": (
                "en_route",
                "arrival",
                "delivery_in_progress",
                "departure",
            ),
            "traffic": ("free", "moderate", "heavy"),
            "nav": ("drive", "wait"),
        }
    )
    # option: forcer 'departure' juste après une fin d'attente (désactivé par défaut)
    force_departure_after_wait: bool = False


# ---- compatibility helper for utils.flexis_export ----
def _extract_schedules_from_ctx(ctx):
    """
    Best-effort extraction of (traffic_profile, weather_timeline) from a
    pipeline/context object. Returns a tuple (traffic_profile, weather_timeline)
    where each item is a list[dict] (may be empty). The function is resilient to
    both dict-like and attribute-like contexts.
    """
    def _deep_get(root, path):
        cur = root
        for key in path:
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                cur = getattr(cur, key, None)
        return cur

    # candidates ordered by most specific first
    traffic_candidates = [
        ["cfg", "flexis", "traffic_profile"],
        ["cfg", "traffic_profile"],
        ["config", "flexis", "traffic_profile"],
        ["config", "traffic_profile"],
        ["flexis", "traffic_profile"],
    ]
    weather_candidates = [
        ["cfg", "flexis", "weather_timeline"],
        ["cfg", "weather_timeline"],
        ["config", "flexis", "weather_timeline"],
        ["config", "weather_timeline"],
        ["flexis", "weather_timeline"],
    ]

    traffic_profile = None
    weather_timeline = None

    for path in traffic_candidates:
        traffic_profile = _deep_get(ctx, path)
        if traffic_profile:
            break
    for path in weather_candidates:
        weather_timeline = _deep_get(ctx, path)
        if weather_timeline:
            break

    # normalize types
    if not isinstance(traffic_profile, list):
        traffic_profile = []
    if not isinstance(weather_timeline, list):
        weather_timeline = []

    return traffic_profile, weather_timeline


class FlexisEnricher:
    """
    Ajoute des colonnes flexis_* au timeline core2.
    Entrée minimale attendue:
      ['timestamp','lat','lon','speed','event','stop_id','t_abs_s','flag_stop','flag_wait']
    Autres colonnes utilisées si dispo:
      ['road_type','osm_highway','distance_m','slope_percent']
    """

    def __init__(self, logger: Optional[logging.Logger] = None, cfg: Optional[FlexisConfig] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.cfg = cfg or FlexisConfig()

    # --------------------------- utils log ---------------------------

    def _dbg(self, msg: str) -> None:
        self.logger.debug("FlexisEnricher: %s", msg)

    # --------------------------- upsert helper ---------------------------

    def _upsert_series(
        self,
        df: pd.DataFrame,
        name: str,
        values: pd.Series,
        dtype: Optional[str] = None,
        fill: Optional[object] = None,
    ) -> None:
        """Insert/update a column, normalise NaN/whitespace and dtypes."""
        s = values.copy()

        # Nettoyage textual NaN/whitespace
        if s.dtype == "O":
            s = s.fillna("").astype(str).map(lambda x: x.strip())
            # homogeneité : remplacer 'nan', 'None' textuels
            s = s.replace({"nan": "", "None": ""})

        if fill is not None:
            s = s.fillna(fill)

        if dtype:
            try:
                if dtype in ("float32", "float64"):
                    s = pd.to_numeric(s, errors="coerce")
                s = s.astype(dtype)
            except Exception:  # garde-fou, mais log
                self._dbg(f"_upsert_series({name}): cast failed -> keep inferred dtype")

        df[name] = s

        non_na = int(s.notna().sum())
        uniq = int(s.nunique(dropna=True))
        head_vals = s.head(3).tolist()
        vc = s.value_counts(dropna=True).head(3).to_dict()
        self._dbg(f"_upsert_series({name}): non_na={non_na}, unique={uniq}")

        # résumé compact
        self._dbg(f"FlexisEnricher summary: {name}: head={head_vals} top={vc}")

    # --------------------------- road type source ---------------------------

    def _choose_road_type_source(self, df: pd.DataFrame) -> str:
        for col in self.cfg.road_type_priority:
            if col in df.columns:
                return col
        return self.cfg.road_type_priority[0]

    # --------------------------- weather ---------------------------

    def _timeline_to_series(self, times: pd.Series, timeline: List[Dict[str, str]], key: str, default: str) -> pd.Series:
        """
        timeline format: [{"from":"08:00","to":"11:00", key: "..."}]
        times is expected in pandas datetime (UTC or local-consistent).
        """
        out = pd.Series(default, index=times.index, dtype="object")
        if not timeline:
            return out

        # on ne regarde que l'heure locale du timestamp fourni
        hhmm = times.dt.strftime("%H:%M")
        for slot in timeline:
            v = slot.get(key, default)
            f = slot.get("from", "00:00")
            t = slot.get("to", "23:59")
            mask = (hhmm >= f) & (hhmm < t)
            out.loc[mask] = v
        return out

    def _compute_weather(self, df: pd.DataFrame) -> pd.Series:
        timeline = (self.cfg.weather_timeline or [])[:]
        if not timeline:
            # fallback inconnu
            return pd.Series("unknown", index=df.index, dtype="object")
        ts = self._ensure_datetime(df)
        return self._timeline_to_series(ts, timeline, "weather", "unknown")

    # --------------------------- traffic ---------------------------

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.Series:
        if "time_utc" in df.columns:
            return pd.to_datetime(df["time_utc"], errors="coerce", utc=True).dt.tz_convert(None)
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"], errors="coerce")
        # fallback via t_abs_s depuis t0 arbitraire
        if "t_abs_s" in df.columns:
            t0 = pd.Timestamp("2020-01-01 00:00:00")
            return t0 + pd.to_timedelta(df["t_abs_s"], unit="s")
        # sinon index
        return pd.to_datetime(df.index, errors="coerce")

    def _compute_traffic_level(self, df: pd.DataFrame, hz: float | None = None) -> pd.Series:
        labels = self.cfg.labels["traffic"]
        # 1) timeline YAML si présente
        if self.cfg.traffic_profile:
            ts = self._ensure_datetime(df)
            s = self._timeline_to_series(ts, self.cfg.traffic_profile, "level", labels[1])  # default moderate
            return s.astype("object")

        # 2) fallback: rolling speed quantiles par type de route
        speed = pd.to_numeric(df.get("speed", pd.Series(index=df.index, dtype="float64"))).fillna(0.0)
        # Choix d'une fenêtre en points, stable vis-à-vis de petites irrégularités temporelles.
        if hz and hz > 0:
            win = int(max(5, round(TRAFFIC_ROLLING_S * hz)))
        elif "t_abs_s" in df.columns:
            # Approxime la fréquence à partir de t_abs_s si possible
            try:
                t = pd.to_numeric(df["t_abs_s"], errors="coerce").astype("float64")
                d = t.diff().dropna()
                d = d[d > 0]
                med = float(d.median()) if len(d) else 0.0
                est_hz = (1.0 / med) if med > 0 else 10.0
            except Exception:
                est_hz = 10.0
            win = int(max(5, round(TRAFFIC_ROLLING_S * est_hz)))
        else:
            # fallback conservateur
            win = 30

        v_roll = speed.rolling(win, min_periods=max(3, win // 3)).median()

        # thresholds dynamiques via quantiles globaux
        q_free = v_roll.quantile(0.75)
        q_heavy = v_roll.quantile(0.30)
        self._dbg(f"traffic fallback: q30={q_heavy:.3f} q75={q_free:.3f}")

        out = pd.Series(labels[1], index=df.index, dtype="object")  # moderate par défaut
        out[v_roll >= q_free] = labels[0]  # free
        out[v_roll <= q_heavy] = labels[2]  # heavy
        return out

    # --------------------------- population density (simple) ---------------------------

    def _compute_population_density(self, df: pd.DataFrame, road_source: str) -> pd.Series:
        """
        Fallback très simple basé sur le type de route.
        (à affiner si une source externe ou grille est disponible)
        """
        rt = df.get(road_source, pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
        # valeurs en cohérence avec ce qu'on a vu dans les logs (pour stabilité)
        mapping = {
            "residential": 3000.0,
            "service": 1200.0,
            "tertiary": 2000.0,
            "secondary": 1500.0,
            "primary": 800.0,
            "motorway": 200.0,
        }
        dens = rt.map(mapping).fillna(1200.0).astype("float32")
        return dens

    # --------------------------- courbure & rayon ---------------------------

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360.0) % 360.0

    def _compute_curve_radius(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lon = pd.to_numeric(df["lon"], errors="coerce")
        # bearing point-à-point
        b = np.zeros(len(df), dtype="float64")
        b[:] = np.nan
        idx = np.arange(len(df))
        idx2 = idx[1:]
        b[1:] = [
            self._bearing_deg(lat.iloc[i - 1], lon.iloc[i - 1], lat.iloc[i], lon.iloc[i])
            for i in idx2
        ]
        # lissage simple
        b_series = pd.Series(b, index=df.index)
        b_series = b_series.bfill().ffill()
        if HEADING_SMOOTH_N > 1:
            b_series = b_series.rolling(HEADING_SMOOTH_N, min_periods=1, center=True).median()

        # delta heading (rad) modulo 180° pour éviter grands sauts
        dtheta_deg = (b_series.diff().fillna(0.0) + 540.0) % 360.0 - 180.0
        dtheta = np.deg2rad(dtheta_deg.astype("float64"))

        # distance inter-points
        dist = pd.Series(0.0, index=df.index, dtype="float64")
        dist.iloc[1:] = [
            self._haversine_m(lat.iloc[i - 1], lon.iloc[i - 1], lat.iloc[i], lon.iloc[i])
            for i in idx2
        ]
        ds = dist.replace(0, np.nan)

        # courbure approx kappa = dtheta/ds
        kappa = (dtheta / ds).replace([np.inf, -np.inf], np.nan)
        # rayon = 1/|kappa|
        radius = pd.Series(STRAIGHT_RADIUS_FALLBACK, index=df.index, dtype="float64")
        valid = kappa.abs() > 0
        radius.loc[valid] = (1.0 / kappa.abs().where(kappa.abs() > 0)).clip(upper=STRAIGHT_RADIUS_FALLBACK)
        radius = radius.fillna(STRAIGHT_RADIUS_FALLBACK)

        is_curve = (radius < CURVE_RADIUS_THRESHOLD_M)
        return radius.astype("float64"), is_curve.astype("bool")

    # --------------------------- machine à états livraison ---------------------------

    def _delivery_fsm(self, df: pd.DataFrame) -> pd.Series:
        """
        en_route → arrival → delivery_in_progress → departure → en_route
        Détection:
          - arrival: proche d'un stop (flag_stop) ou vitesse < v_arrive et flag_wait==False
          - in_progress: flag_wait==True ou vitesse quasi nulle pendant un certain temps
          - departure: reprise vitesse > v_depart sur quelques secondes
        Le tout sans s'appuyer sur un éventuel is_service.
        """
        labels = self.cfg.labels["delivery"]
        v = pd.to_numeric(df.get("speed", pd.Series(index=df.index, dtype="float64")), errors="coerce").fillna(0.0)

        # paramètres heuristiques
        v_arrive = 1.2    # m/s ~ 4.3 km/h
        v_stop = 0.3      # m/s
        v_depart = 1.5    # m/s  (plus permissif pour capter la reprise)
        # fenêtre ~1s si la fréquence est connue, sinon valeur par défaut
        inferred_hz = getattr(self, "_hz", None)
        if inferred_hz and inferred_hz > 0:
            win_pts = int(max(5, round(1.0 * inferred_hz)))
        else:
            win_pts = 7

        # prox stop (au sens core2)
        near_stop = df.get("flag_stop", pd.Series(False, index=df.index)).fillna(False).astype(bool)

        # attente explicite
        waiting = df.get("flag_wait", pd.Series(False, index=df.index)).fillna(False).astype(bool)

        # glissements simples
        v_med = v.rolling(win_pts, min_periods=1).median()

        states = np.empty(len(df), dtype=object)
        state = labels[0]  # en_route initial

        for i in range(len(df)):
            if state == labels[0]:  # en_route
                if near_stop.iloc[i] and v_med.iloc[i] < v_arrive:
                    state = labels[1]  # arrival
            elif state == labels[1]:  # arrival
                if waiting.iloc[i] or v_med.iloc[i] <= v_stop:
                    state = labels[2]  # delivery_in_progress
                elif v_med.iloc[i] > v_depart and not near_stop.iloc[i]:
                    # pas d'arrêt finalement
                    state = labels[0]
            elif state == labels[2]:  # delivery_in_progress
                if v_med.iloc[i] > v_depart and not waiting.iloc[i]:
                    state = labels[3]  # departure
            elif state == labels[3]:  # departure
                if v_med.iloc[i] > v_depart and not near_stop.iloc[i]:
                    state = labels[0]  # retour en route
                elif waiting.iloc[i] or v_med.iloc[i] <= v_stop:
                    # re-bascule si encore en service
                    state = labels[2]
            states[i] = state

        out = pd.Series(states, index=df.index, dtype="object")

        # si plusieurs arrêts distincts existent via stop_id, forcer transitions locales
        if "stop_id" in df.columns:
            stop_change = df["stop_id"].astype(str).fillna("").ne(df["stop_id"].astype(str).fillna("")).astype(bool)
            # un changement de stop force l'état arrival au prochain point si on est near_stop
            out.loc[stop_change & near_stop] = labels[1]

        # Option: forcer 'departure' après la fin d'une attente si on s'éloigne du stop
        if getattr(self.cfg, "force_departure_after_wait", False):
            horizon = 5  # points à regarder devant
            for i in range(1, len(df)):
                # fin d'une période d'attente
                if waiting.iloc[i - 1] and not waiting.iloc[i]:
                    end = min(len(df), i + horizon)
                    for j in range(i, end):
                        if (v_med.iloc[j] > v_depart) and (not near_stop.iloc[j]):
                            out.iloc[j] = labels[3]  # departure
                            break

        return out

    # --------------------------- jour/nuit ---------------------------

    def _compute_night(self, df: pd.DataFrame) -> pd.Series:
        ts = self._ensure_datetime(df)
        hh = ts.dt.hour.fillna(12).astype(int)
        night = (hh < 6) | (hh >= 21)
        return night.astype("bool")

    # --------------------------- nav_event (simple) ---------------------------

    def _compute_nav_event(self, df: pd.DataFrame) -> pd.Series:
        labels = self.cfg.labels["nav"]
        waiting = df.get("flag_wait", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        nav = pd.Series(labels[0], index=df.index, dtype="object")
        nav.loc[waiting] = labels[1]
        return nav

    # --------------------------- public API ---------------------------

    def run(self, ctx_or_df):
        """
        API tolérante:
          - si on reçoit un DataFrame => retourne un DataFrame enrichi
          - si on reçoit un contexte (core2 Context-like) => lit un DF dans
            ctx.df / ctx.timeline / ctx.data, écrit les colonnes et retourne le ctx
        """
        # Déduire le contexte et le dataframe d'entrée
        ctx = None
        if isinstance(ctx_or_df, pd.DataFrame):
            df_in = ctx_or_df
        else:
            ctx = ctx_or_df
            # récupérer un DataFrame plausible depuis le contexte
            df_in = None
            for attr in ("df", "timeline", "data"):
                if hasattr(ctx, attr):
                    candidate = getattr(ctx, attr)
                    if isinstance(candidate, pd.DataFrame):
                        df_in = candidate
                        break
            if df_in is None:
                raise AttributeError("FlexisEnricher: aucun DataFrame trouvé dans le contexte (attendu ctx.df/ctx.timeline)")

            # si la config n'a pas été fournie au constructeur, tenter extraction depuis ctx
            if (not self.cfg.traffic_profile) or (not self.cfg.weather_timeline):
                try:
                    tp, wt = _extract_schedules_from_ctx(ctx)
                    if not self.cfg.traffic_profile:
                        self.cfg.traffic_profile = tp
                    if not self.cfg.weather_timeline:
                        self.cfg.weather_timeline = wt
                except Exception:
                    # best-effort seulement
                    pass

        df = df_in.copy()

        # --- Diagnostics & Hz inference (sans mutation de l'axe temps)
        self._hz = _infer_hz(df)
        _log_time_axis_health(self.logger, df, self._hz)

        # journalisation d'entrée
        cols = list(df.columns)
        self._dbg(f"input columns: {cols}")

        # choisir la source type de route
        road_src = self._choose_road_type_source(df)
        available = [c for c in (road_src, "osm_highway") if c in df.columns]
        self._dbg(f"available road type columns: {available}")
        self._dbg(f"road_type source: {road_src}")

        # trafic & météo
        traffic = self._compute_traffic_level(df, hz=self._hz)
        weather = self._compute_weather(df)

        # night
        night = self._compute_night(df)

        # densité
        density = self._compute_population_density(df, road_src)

        # rayon de courbure + flag
        try:
            radius_m, is_curve = self._compute_curve_radius(df)
        except Exception as e:
            self._dbg(f"curve radius failed: {e!r}; fallback radius={STRAIGHT_RADIUS_FALLBACK}")
            radius_m = pd.Series(STRAIGHT_RADIUS_FALLBACK, index=df.index, dtype="float64")
            is_curve = pd.Series(False, index=df.index, dtype="bool")

        # nav & livraison
        nav_event = self._compute_nav_event(df)
        delivery_status = self._delivery_fsm(df)

        # type de route nominal (copie)
        road_type_series = df.get(road_src, pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
        # normalisation soft
        road_type_series = road_type_series.replace({"": "unknown"})

        # ~~~ UPserts (normalisés / typés) ~~~
        self._upsert_series(df, "flexis_road_type", road_type_series, dtype="object")
        self._upsert_series(df, "flexis_road_curve_radius_m", radius_m, dtype="float64")
        self._upsert_series(df, "flexis_is_curve", is_curve, dtype="bool")
        self._upsert_series(df, "flexis_nav_event", nav_event, dtype="object")
        self._upsert_series(df, "flexis_weather", weather, dtype="object")
        self._upsert_series(df, "flexis_night", night, dtype="bool")
        self._upsert_series(df, "flexis_traffic_level", traffic, dtype="object")
        self._upsert_series(df, "flexis_delivery_status", delivery_status, dtype="object")
        # infra / driver events: si absents, insérer blancs
        infra = df.get("flexis_infra_event", pd.Series("", index=df.index, dtype="object"))
        driver = df.get("flexis_driver_event", pd.Series("", index=df.index, dtype="object"))
        self._upsert_series(df, "flexis_infra_event", infra, dtype="object")
        self._upsert_series(df, "flexis_driver_event", driver, dtype="object")
        # densité (float32 pour économiser la taille de fichier)
        self._upsert_series(df, "flexis_population_density_km2", density, dtype="float32")

        # récap compact
        out_cols = [
            "flexis_road_type",
            "flexis_road_curve_radius_m",
            "flexis_is_curve",
            "flexis_nav_event",
            "flexis_weather",
            "flexis_night",
            "flexis_traffic_level",
            "flexis_delivery_status",
            "flexis_infra_event",
            "flexis_driver_event",
            "flexis_population_density_km2",
        ]
        self._dbg(f"run: flexis_cols in out: {out_cols}")
        for c in out_cols:
            s = df[c]
            self._dbg(f"  {c}: non_na={int(s.notna().sum())}, unique={int(s.nunique(dropna=True))}")

        # Écriture dans le contexte si besoin
        if ctx is not None:
            for attr in ("df", "timeline", "data"):
                if hasattr(ctx, attr) and isinstance(getattr(ctx, attr), pd.DataFrame):
                    setattr(ctx, attr, df)
                    break
            return ctx

        return df