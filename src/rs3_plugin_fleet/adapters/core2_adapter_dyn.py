# -*- coding: utf-8 -*-
"""
Adapter dynamique → implémentation core2.*
Expose l'API minimale attendue par rs3_plugin_fleet.pipeline.builder :
  - get_pipeline_cls()
  - get_result_cls()
  - get_context_cls()
  - build_default_stages(cfg)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import importlib


# ---------------------------------------------------------------------------
# Utils

def _resolve(dotted: str):
    """
    Importe 'pkg.mod:Class' OU 'pkg.mod.Class' et retourne l'objet visé.
    """
    if ":" in dotted:
        mod_name, attr = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    return obj


def _try_resolve(*candidates: str):
    """
    Essaie plusieurs chemins d'import et retourne le premier qui marche.
    """
    last_err = None
    for dotted in candidates:
        try:
            return _resolve(dotted)
        except Exception as e:  # garde la trace mais continue à essayer
            last_err = e
            continue
    if last_err:
        raise last_err
    raise ImportError("No candidates provided to _try_resolve()")


# ---------------------------------------------------------------------------
# API demandée par builder.py

def get_pipeline_cls():
    """Retourne la classe du pipeline core2 (simulateur)."""
    # ex: core2.pipeline.PipelineSimulator
    return _try_resolve("core2.pipeline:PipelineSimulator")


def get_result_cls():
    """
    Retourne la classe Result utilisée par le pipeline core2 pour emballer
    les retours de stage. On préfère rs3_contracts.api.Result si présent.
    """
    try:
        return _resolve("rs3_contracts.api:Result")
    except Exception:
        # Fallback léger : petit shim compatible (bool -> (ok,msg))
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


def get_context_cls():
    """Retourne la classe Context core2."""
    return _try_resolve("core2.context:Context")


def _instantiate(cls, cfg_section: Optional[Dict[str, Any]] = None):
    """Instancie une classe avec une section de config si possible."""
    cfg_section = cfg_section or {}
    try:
        return cls(**cfg_section)
    except TypeError:
        # Certains stages n’ont pas de kwargs
        return cls()


def build_default_stages(cfg: Dict[str, Any]) -> List[Any]:
    """
    Construit la pipeline "par défaut" de core2 (sans patchs plugin).
    On instancie chaque stage avec sa section de config si elle existe
    dans `cfg` (e.g. cfg["legs_plan"], cfg["speed_sync"], etc.).
    """
    stages: List[Any] = []

    # Résolution des classes core2 avec fallback pour les rename de modules
    LegsPlan        = _try_resolve("core2.stages.legs_plan:LegsPlan")
    LegsRoute       = _try_resolve("core2.stages.legs_route:LegsRoute")
    LegsStitch      = _try_resolve("core2.stages.legs_stitch:LegsStitch")
    RoadEnricher    = _try_resolve("core2.stages.road_enricher:RoadEnricher")
    GeoSpikeFilter  = _try_resolve("core2.stages.geo_spike_filter:GeoSpikeFilter")
    LegsRetimer     = _try_resolve("core2.stages.legs_retimer:LegsRetimer")
    StopWaitInjector= _try_resolve("core2.stages.stopwait_injector:StopWaitInjector")
    StopSmoother    = _try_resolve("core2.stages.stop_smoother:StopSmoother")

    # lockers : anciens chemins, puis nouveaux
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

    IMUProjector    = _try_resolve(
        "core2.stages.imu_projector:IMUProjector",
        "core2.stages.imu_projector:IMUProjector",  # alias stable
    )
    NoiseInjector   = _try_resolve("core2.stages.noise_injector:NoiseInjector")
    SpeedSync       = _try_resolve("core2.stages.speed_sync:SpeedSync")
    Validators      = _try_resolve("core2.stages.validators:Validators")
    Exporter        = _try_resolve("core2.stages.exporter:Exporter")

    # Récupère la section de config si fournie
    c = lambda key: (cfg.get(key) or {}) if isinstance(cfg.get(key), dict) else {}

    # Ordre par défaut observé dans tes logs
    stages.append(_instantiate(LegsPlan,         c("legs_plan")))
    stages.append(_instantiate(LegsRoute,        c("legs_route")))
    stages.append(_instantiate(LegsStitch,       c("legs_stitch")))
    stages.append(_instantiate(RoadEnricher,     c("road_enricher")))
    stages.append(_instantiate(GeoSpikeFilter,   c("geo_spike_filter")))
    stages.append(_instantiate(LegsRetimer,      c("legs_retimer")))
    stages.append(_instantiate(StopWaitInjector, c("stopwait_injector")))
    stages.append(_instantiate(StopSmoother,     c("stop_smoother")))
    stages.append(_instantiate(InitialStopLocker, c("initial_stop_locker")))
    stages.append(_instantiate(MidStopsLocker,    c("mid_stops_locker")))
    stages.append(_instantiate(FinalStopLocker,   c("final_stop_locker")))
    stages.append(_instantiate(IMUProjector,     c("imu_projector")))
    stages.append(_instantiate(NoiseInjector,    c("noise_injector")))
    stages.append(_instantiate(SpeedSync,        c("speed_sync")))
    stages.append(_instantiate(Validators,       c("validators")))
    stages.append(_instantiate(Exporter,         c("exporter")))

    return stages


__all__ = [
    "get_pipeline_cls",
    "get_result_cls",
    "get_context_cls",
    "build_default_stages",
]