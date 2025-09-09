#
# -*- coding: utf-8 -*-
"""
Pipeline builder — assemble core pipeline + altitude plugin + fillers/finalizers.
Ce module ne dépend pas directement de core2.* : il passe par l'adapter dynamique.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import importlib

from rs3_plugin_fleet.plugin_discovery.altitude_loader import load_altitude_enricher
from rs3_plugin_fleet.utils.ctx_access import CtxAccessor
from rs3_plugin_fleet.stages.flexis_features import FlexisFeaturesStage

ADAPTER_MODULE = os.environ.get("RS3_IMPL_ADAPTER", "rs3_plugin_fleet.adapters.core2_adapter_dyn")

def _load_impl():
    mod = importlib.import_module(ADAPTER_MODULE)
    required = ["get_pipeline_cls", "get_result_cls", "get_context_cls", "build_default_stages"]
    for attr in required:
        if not hasattr(mod, attr):
            raise ImportError(f"Adapter '{ADAPTER_MODULE}' invalide, attribut manquant: {attr}")
    return mod

_impl = _load_impl()

def _split_for_tail_insertion(stages: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    head: List[Any] = []
    validators: List[Any] = []
    exporters: List[Any] = []

    def _is_validator(obj: Any) -> bool:
        if getattr(obj, "is_validator", False):
            return True
        n = type(obj).__name__.lower()
        return any(k in n for k in ("validator","validate","check","sanity","consistency"))

    def _is_exporter(obj: Any) -> bool:
        if getattr(obj, "is_exporter", False):
            return True
        n = type(obj).__name__.lower()
        return any(k in n for k in ("export","writer","report","dump","save"))

    for s in stages:
        if _is_validator(s):
            validators.append(s)
        elif _is_exporter(s):
            exporters.append(s)
        else:
            head.append(s)
    return head, validators, exporters

def _maybe_altitude_stages(cfg: Dict[str, Any]) -> List[Any]:
    alt_cfg = cfg.get("altitude", {}) or {}
    enabled = bool(alt_cfg.get("enabled", True))
    if not enabled:
        return []

    if not alt_cfg.get("base_url") and not alt_cfg.get("base"):
        alt_cfg["base_url"] = os.environ.get("RS3_ALTITUDE_BASE", "http://localhost:5004")

    alt_class = alt_cfg.get("class")
    if isinstance(alt_class, str) and ":" in alt_class:
        os.environ["RUNNER_ALTITUDE_CLASS"] = alt_class

    if "timeout_s" in alt_cfg and "timeout" not in alt_cfg:
        alt_cfg["timeout"] = alt_cfg["timeout_s"]
    alt_cfg.setdefault("timeout", 30.0)
    if "base" in alt_cfg and "base_url" not in alt_cfg:
        alt_cfg["base_url"] = alt_cfg["base"]
    _t = alt_cfg.get("timeout")
    alt_cfg.setdefault("timeout_s", _t)
    alt_cfg.setdefault("read_timeout", _t)
    alt_cfg.setdefault("request_timeout", _t)
    alt_cfg.setdefault("connect_timeout", _t)

    print(f"[ALT] Using base_url={alt_cfg.get('base_url') or alt_cfg.get('base')}, timeout={alt_cfg.get('timeout')}s")

    AltitudeEnricher = load_altitude_enricher()
    if AltitudeEnricher is None:
        print("[WARN] Plugin altitude indisponible — on continue sans.")
        return []
    try:
        obj = AltitudeEnricher(**alt_cfg)
    except Exception:
        obj = AltitudeEnricher()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj] if obj is not None else []

def build_pipeline(cfg: Dict[str, Any]):
    Pipeline = _impl.get_pipeline_cls()
    CoreResult = _impl.get_result_cls()

    impl_stages = _impl.build_default_stages(cfg)
    head, validators, exporters = _split_for_tail_insertion(impl_stages)

    stages: List[Any] = []
    stages += head
    stages += _maybe_altitude_stages(cfg)         # plugin altitude

    _tail_s = float(cfg.get("tail_zero_seconds", 8.0))
    _hz = float(cfg.get("hz", 10.0))

    stages.append(FlexisFeaturesStage(hz=_hz))
    stages += validators

    # --- Timestamp re-anchor: fix accidental epoch(1970) dates while preserving time-of-day
    def _reanchor_ts(df, ctx):
        if df is None or "timestamp" not in df.columns:
            return df
        ts = df["timestamp"]
        try:
            import pandas as pd  # local import to avoid hard dependency at import time
            ts_dt = pd.to_datetime(ts, utc=True, errors="coerce")
        except Exception:
            return df
        if ts_dt.isna().all():
            return df
        # if a majority of rows are in 1970, we re-anchor
        year = ts_dt.dt.year
        if (year == 1970).sum() < 0.8 * len(ts_dt):
            return df
        # choose anchor day
        base = cfg.get("run_start_iso")
        try:
            base_dt = pd.to_datetime(base, utc=True) if base else pd.Timestamp.utcnow().normalize().tz_localize("UTC")
        except Exception:
            base_dt = pd.Timestamp.utcnow().normalize().tz_localize("UTC")
        # keep time-of-day: (ts - midnight) + base_midnight
        tod = (ts_dt - ts_dt.dt.normalize())
        df["timestamp"] = base_dt + tod
        return df

    try:
        stages.append(CtxAccessor(_reanchor_ts))
    except Exception:
        pass

    stages += exporters

    # ---- DEBUG: log assembled pipeline & warn if key core stages are missing
    try:
        _names = [type(s).__name__ for s in stages]
        print("[PIPELINE] Stages before final clamp:", " → ".join(_names))
        _must = {"SpeedSync", "RoadEnricher", "GeoSpikeFilter", "LegsRetimer", "ImuProjector", "Exporter"}
        _missing = [n for n in _must if n not in _names]
        if _missing:
            print("[WARN] Missing expected core stages:", ", ".join(_missing),
                  "— check adapter.build_default_stages(cfg) and YAML 'stages' usage.")
    except Exception as _e:
        print("[DEBUG] Unable to introspect stages:", _e)
    # ---- /DEBUG

    # Remove FinalStopLocker to avoid artificial tail wait/lock at end of run
    stages = [s for s in stages if type(s).__name__ != "FinalStopLocker"]

    return Pipeline(stages)