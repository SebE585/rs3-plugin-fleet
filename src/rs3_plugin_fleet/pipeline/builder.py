# -*- coding: utf-8 -*-
"""
Dynamic adapter for RS3 core2 — standalone (no import from pipeline.builder).
Exposes the attributes expected by rs3_plugin_fleet.pipeline.builder:
  - get_pipeline_cls()
  - get_context_cls()
  - get_result_cls()
  - build_default_stages(cfg)
  - build_pipeline_and_ctx(cfg, sim_cfg=None, config_path=None)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import importlib

# ----------------------------
# Minimal dynamic resolvers
# ----------------------------
def _resolve(dotted: str):
    """
    Import 'pkg.mod:Attr' or 'pkg.mod.Attr' and return the symbol.
    """
    if ":" in dotted:
        mod_name, attr = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)

def _try_resolve(*candidates: str):
    last_err = None
    for dotted in candidates:
        try:
            return _resolve(dotted)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise ImportError("No candidates provided to _try_resolve()")

# ----------------------------
# Public API (used by adapter and builder)
# ----------------------------
def get_pipeline_cls():
    """Return core2 pipeline simulator class."""
    return _try_resolve("core2.pipeline:PipelineSimulator")

def get_context_cls():
    """Return core2 Context class."""
    return _try_resolve("core2.context:Context")

def get_result_cls():
    """
    Return the Result class used by the contracts layer if available,
    otherwise provide a tiny compatible shim exposing .ok and .msg.
    """
    try:
        return _resolve("rs3_contracts.api:Result")
    except Exception:
        class _Result(tuple):  # type: ignore
            def __new__(cls, val):
                if isinstance(val, tuple) and len(val) == 2:
                    return super().__new__(cls, val)
                return super().__new__(cls, (bool(val), ""))
            @property
            def ok(self) -> bool:
                return bool(self[0])
            @property
            def msg(self) -> str:
                return str(self[1])
        return _Result

# ----------------------------
# Stage composition helpers
# ----------------------------
def _instantiate(cls, section: Optional[Dict[str, Any]] = None):
    section = section or {}
    try:
        return cls(**section)
    except TypeError:
        return cls()

def build_default_stages(cfg: Dict[str, Any]) -> List[Any]:
    """
    Build the default stage list observed in RS3 core2, instantiating each
    with its config section if present in `cfg`.
    """
    stages: List[Any] = []

    LegsPlan         = _try_resolve("core2.stages.legs_plan:LegsPlan")
    LegsRoute        = _try_resolve("core2.stages.legs_route:LegsRoute")
    LegsStitch       = _try_resolve("core2.stages.legs_stitch:LegsStitch")
    RoadEnricher     = _try_resolve("core2.stages.road_enricher:RoadEnricher")
    GeoSpikeFilter   = _try_resolve("core2.stages.geo_spike_filter:GeoSpikeFilter")
    LegsRetimer      = _try_resolve("core2.stages.legs_retimer:LegsRetimer")
    StopWaitInjector = _try_resolve("core2.stages.stopwait_injector:StopWaitInjector")
    StopSmoother     = _try_resolve("core2.stages.stop_smoother:StopSmoother")

    InitialStopLocker = _try_resolve(
        "core2.stages.stop_lockers:InitialStopLocker",
        "core2.stages.initial_stop_locker:InitialStopLocker",
    )
    MidStopsLocker = _try_resolve(
        "core2.stages.stop_lockers:MidStopsLocker",
        "core2.stages.mid_stops_locker:MidStopsLocker",
    )
    FinalStopLocker = _try_resolve(
        "core2.stages.stop_lockers:FinalStopLocker",
        "core2.stages.final_stop_locker:FinalStopLocker",
    )

    IMUProjector   = _try_resolve("core2.stages.imu_projector:IMUProjector")
    NoiseInjector  = _try_resolve("core2.stages.noise_injector:NoiseInjector")
    SpeedSync      = _try_resolve("core2.stages.speed_sync:SpeedSync")
    Validators     = _try_resolve("core2.stages.validators:Validators")
    Exporter       = _try_resolve("core2.stages.exporter:Exporter")

    c = lambda key: (cfg.get(key) or {}) if isinstance(cfg.get(key), dict) else {}

    stages.append(_instantiate(LegsPlan,          c("legs_plan")))
    stages.append(_instantiate(LegsRoute,         c("legs_route")))
    stages.append(_instantiate(LegsStitch,        c("legs_stitch")))
    stages.append(_instantiate(RoadEnricher,      c("road_enricher")))
    stages.append(_instantiate(GeoSpikeFilter,    c("geo_spike_filter")))
    stages.append(_instantiate(LegsRetimer,       c("legs_retimer")))
    stages.append(_instantiate(StopWaitInjector,  c("stopwait_injector")))
    stages.append(_instantiate(StopSmoother,      c("stop_smoother")))
    stages.append(_instantiate(InitialStopLocker, c("initial_stop_locker")))
    stages.append(_instantiate(MidStopsLocker,    c("mid_stops_locker")))
    stages.append(_instantiate(FinalStopLocker,   c("final_stop_locker")))
    stages.append(_instantiate(IMUProjector,      c("imu_projector")))
    stages.append(_instantiate(NoiseInjector,     c("noise_injector")))
    stages.append(_instantiate(SpeedSync,         c("speed_sync")))
    stages.append(_instantiate(Validators,        c("validators")))
    stages.append(_instantiate(Exporter,          c("exporter")))
    return stages

# ----------------------------
# High-level builder (standalone)
# ----------------------------
def build_pipeline_and_ctx(
    cfg: Dict[str, Any],
    sim_cfg: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Compose a runnable Pipeline + Context without importing rs3_plugin_fleet.pipeline.builder.
    """
    PipelineCls = get_pipeline_cls()
    ContextCls  = get_context_cls()

    base_cfg = sim_cfg if isinstance(sim_cfg, dict) else (cfg if isinstance(cfg, dict) else {})
    stages = build_default_stages(base_cfg)

    # Instantiate pipeline and inject stages
    pipeline = None
    try:
        pipeline = PipelineCls(stages=stages)
    except Exception:
        try:
            pipeline = PipelineCls(base_cfg)
        except Exception:
            pipeline = PipelineCls()

    if hasattr(pipeline, "set_stages") and callable(getattr(pipeline, "set_stages")):
        try:
            pipeline.set_stages(stages)
        except Exception:
            pass
    elif hasattr(pipeline, "stages"):
        try:
            setattr(pipeline, "stages", stages)
        except Exception:
            pass

    # Build context
    try:
        ctx = ContextCls(base_cfg)
    except Exception:
        try:
            ctx = ContextCls()
            if hasattr(ctx, "set_config"):
                try:
                    ctx.set_config(base_cfg)
                except Exception:
                    pass
            else:
                setattr(ctx, "config", base_cfg)
        except Exception as e:
            raise RuntimeError(f"[ADAPTER] Unable to instantiate Context: {e}") from e

    # Sanity checks
    if not hasattr(pipeline, "run") or not callable(getattr(pipeline, "run", None)):
        raise RuntimeError("[ADAPTER] Composed pipeline is not runnable (.run() missing)")
    if not stages or any(isinstance(s, (str, dict)) for s in stages):
        raise RuntimeError("[ADAPTER] Invalid stages after dynamic composition")

    return pipeline, ctx

__all__ = [
    "get_pipeline_cls",
    "get_context_cls",
    "get_result_cls",
    "build_default_stages",
    "build_pipeline_and_ctx",
]