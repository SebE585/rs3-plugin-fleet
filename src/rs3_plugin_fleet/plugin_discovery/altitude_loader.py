from __future__ import annotations
import os, importlib
try:
    from importlib import metadata as im
except Exception:  # pragma: no cover
    import importlib_metadata as im  # type: ignore

CANDIDATES = [
    "core2.stages.altitude_enricher:AltitudeEnricher",
    "core2.plugins.altitude:AltitudeEnricher",
    "plugins.altitude:AltitudeEnricher",
    "core2.stages.altitude_enricher:AltitudeStage",
    "core2.plugins.altitude:AltitudeStage",
    "plugins.altitude:AltitudeStage",
    "core2.plugins.altitude_stage:AltitudeStage",
    "plugins.altitude_stage:AltitudeStage",
    # External AGPL plugin
    "rs3_plugin_altitude_agpl.altitude_enricher:AltitudeStage",
    "rs3_plugin_altitude_agpl.altitude_enricher:AltitudeEnricher",
]

def load_altitude_enricher():
    # Env override (YAML peut remplir RUNNER_ALTITUDE_CLASS)
    env_ref = os.environ.get("RS3_ALTITUDE_CLASS") or os.environ.get("RUNNER_ALTITUDE_CLASS")
    for ref in ([env_ref] if env_ref else [] ) + CANDIDATES:
        if not ref: continue
        mod, _, attr = ref.partition(":")
        try:
            m = importlib.import_module(mod)
            return getattr(m, attr)
        except Exception:
            pass

    # Entry points
    try:
        eps = im.entry_points()
        group_eps = eps.select(group="rs3.plugins") if hasattr(eps, "select") \
                     else [ep for ep in eps if getattr(ep, "group", None) == "rs3.plugins"]
        for ep in group_eps:
            name = (getattr(ep, "name", "") or "").lower()
            value = (getattr(ep, "value", "") or "").lower()
            if "altitude" in name or "altitude" in value:
                try:
                    return ep.load()
                except Exception:
                    pass
    except Exception:
        pass
    return None