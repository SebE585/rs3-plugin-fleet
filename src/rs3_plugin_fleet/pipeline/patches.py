# -*- coding: utf-8 -*-
from __future__ import annotations

def _extract_stage_names(pipeline) -> list[str]:
    """
    Heuristiques pour extraire les noms de stages sans dépendre du core :
    - pipeline.stage_names()
    - keys(pipeline.stages) si dict
    - [s.name] si pipeline.stages est une liste et que s.name existe
    - sinon classes des objets ([type(s).__name__])
    """
    try:
        if hasattr(pipeline, "stage_names"):
            names = list(pipeline.stage_names())  # type: ignore[call-arg]
            if names:
                return names
    except Exception:
        pass

    try:
        stages = getattr(pipeline, "stages", None)
        if stages is None:
            return []
        # dict-like
        if isinstance(stages, dict):
            return list(stages.keys())
        # list/iterable
        seq = list(stages)
        if not seq:
            return []
        # try .name
        names = []
        for s in seq:
            n = getattr(s, "name", None)
            if isinstance(n, str) and n:
                names.append(n)
            else:
                names.append(type(s).__name__)
        return names
    except Exception:
        return []

NEEDED = [
    "LegsPlan","LegsRoute","LegsStitch","RoadEnricher",
    "GeoSpikeFilter","LegsRetimer",
    "StopWaitInjector","StopSmoother",
    "InitialStopLocker","MidStopsLocker","FinalStopLocker",
    "IMUProjector","NoiseInjector","SpeedSync","Validators","Exporter"
]

def patch_pipeline_for_fleet(pipeline, ctx, *, with_altitude: bool, with_flexis_export: bool):
    """
    Manipulation par protocole (noms/ops génériques) — aucun import du core.
    Tolérant aux variations d'API : on n'échoue pas si on ne peut pas lister les stages.
    """
    names = set(_extract_stage_names(pipeline))
    if not names:
        print("[WARN] Impossible d’inspecter les stages du pipeline (API différente). "
              "Je continue sans vérification stricte de présence.")

    else:
        missing = [s for s in NEEDED if s not in names]
        if missing:
            # On n'arrête plus : on informe et on continue (le YAML/ builder peut injecter au run)
            print(f"[WARN] Stages non détectés (peut-être normal selon l’API): {missing}")

    # Altitude optionnelle : insertion après RoadEnricher si possible
    if with_altitude and "AltitudeEnricher" not in names:
        if hasattr(pipeline, "insert_after"):
            try:
                pipeline.insert_after("RoadEnricher", "AltitudeEnricher")
                print("[PATCH] AltitudeEnricher inséré après RoadEnricher")
            except Exception as e:
                print(f"[WARN] Insertion AltitudeEnricher impossible ({e}). "
                      "Place-le via YAML si nécessaire.")
        else:
            print("[WARN] API insert_after absente — configure AltitudeEnricher via YAML.")

    # Export Flexis optionnel
    if with_flexis_export:
        if hasattr(pipeline, "set_exporter"):
            try:
                pipeline.set_exporter("rs3_plugin_fleet.utils.flexis_export:FlexisExporter")
                print("[PATCH] Exporter Flexis activé via set_exporter()")
            except Exception as e:
                print(f"[WARN] set_exporter a échoué ({e}) — essai replace_stage.")
        if hasattr(pipeline, "replace_stage"):
            try:
                from rs3_plugin_fleet.utils.flexis_export import FlexisExporter
                cfg = getattr(ctx, "config", None) or {}
                pipeline.replace_stage("Exporter", FlexisExporter(cfg=cfg.get("exporter", {})))
                print("[PATCH] Exporter remplacé par FlexisExporter via replace_stage()")
            except Exception as e:
                print(f"[WARN] replace_stage('Exporter', FlexisExporter) a échoué ({e}). "
                      "Configure via YAML si nécessaire.")
        elif not hasattr(pipeline, "set_exporter"):
            print("[WARN] Ni set_exporter ni replace_stage — impossible d’activer Flexis automatiquement.")
    return pipeline