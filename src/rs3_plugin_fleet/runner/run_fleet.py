# -*- coding: utf-8 -*-
"""
Runner Fleet — lance une simulation multi-profils à partir d'un YAML.

- Normalise ctx.sim_start (UTC) TRÈS tôt → plus de dates 1970.
- Injecte AltitudeStage si dispo (sinon fallback post-enrich via FlexisExporter).
- Maintient Exporter core + ajoute FlexisExporter (dernier).
- Filets post-run : export_flexis() + rapport Flexis HTML.
- Coupage d'un éventuel arrêt terminal artificiel (close_tail_stop).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Dict

import pandas as pd
import yaml

from rs3_plugin_fleet.utils.time_utils import (
    ensure_export_defaults,
    ensure_sim_start_on_ctx,
)
from rs3_plugin_fleet.pipeline.patches import (
    patch_pipeline_for_flexis,
    call_export_flexis_post,
    call_flexis_report_post,
)


# ---------- IO ---------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- Builder/adapter discovery ----------------------------------------
def _get_builder():
    try:
        import rs3_plugin_fleet.adapters.core2_adapter_dyn as adapter
        if hasattr(adapter, "build_pipeline_and_context"):
            return adapter.build_pipeline_and_context  # type: ignore
        if all(hasattr(adapter, n) for n in ("get_pipeline_cls", "get_context_cls", "build_default_stages")):
            def _from_adapter(cfg):
                PipelineCls = adapter.get_pipeline_cls()
                ContextCls = adapter.get_context_cls()
                stages = adapter.build_default_stages(cfg)
                pipeline = PipelineCls(stages)
                try: ctx = ContextCls(cfg)
                except TypeError: ctx = ContextCls()
                try: setattr(ctx, "config", cfg)
                except Exception: pass
                return pipeline, ctx
            return _from_adapter
    except Exception:
        pass
    try:
        from rs3_plugin_fleet.pipeline.builder import build_pipeline_and_context  # type: ignore
        return build_pipeline_and_context
    except Exception:
        pass
    raise ImportError("Impossible d'importer build_pipeline_and_context.")


# ---------- main --------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RS3 Fleet Runner")
    parser.add_argument("-c", "--config", required=True, help="Chemin YAML (ex: coin-coin-delivery.yaml)")
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    profile_name = cfg.get("client") or cfg.get("name") or cfg.get("fleet", {}).get("name")
    profile_info = cfg.get("profile") or cfg.get("fleet", {}).get("profile")
    if profile_name:
        print(f"[RUN] {profile_name}" + (f" ← profil={profile_info}" if profile_info else ""))

    build = _get_builder()

    def _expand_vehicle_run_cfg(cfg_root: Dict[str, Any], vehicle: Dict[str, Any]) -> Dict[str, Any]:
        run: Dict[str, Any] = {}
        profiles = cfg_root.get("profiles", {}) or {}
        v_profile = vehicle.get("profile")
        prof = profiles.get(v_profile, {}) if v_profile else {}
        run.update(deepcopy(prof))
        # hériter de la racine si non défini par le profil
        for k in (
            "osrm","speed_sync","road_enrich","geo_spike_filter",
            "legs_retimer","stop_wait_injector","stop_smoother",
            "validators","exporter","altitude","flexis","stages"
        ):
            if k in cfg_root and k not in run:
                run[k] = deepcopy(cfg_root[k])
        run["vehicle_id"] = vehicle.get("id")
        run["stops"] = deepcopy(vehicle.get("stops", []))
        run["name"] = cfg_root.get("client") or cfg_root.get("name") or run.get("name") or run.get("vehicle_id")
        run["profile"] = v_profile or run.get("profile")
        if "start_at" not in run and "start_at" in cfg_root:
            run["start_at"] = cfg_root["start_at"]
        out_dir = (
            cfg_root.get("output", {}).get("dir")
            or cfg_root.get("export", {}).get("dir")
            or run.get("outdir")
        )
        if out_dir:
            run.setdefault("output", {})
            run["output"]["dir"] = out_dir
        return run

    vehicles = cfg.get("vehicles") or []
    runs = [ _expand_vehicle_run_cfg(cfg, v) for v in vehicles ] if vehicles else [cfg]

    exit_code = 0
    for i, run_cfg in enumerate(runs, start=1):
        name = run_cfg.get("vehicle_id") or run_cfg.get("name") or f"run-{i:02d}"
        prof = run_cfg.get("profile")
        if name and prof: print(f"[RUN] {name} ← profil={prof}")
        elif name: print(f"[RUN] {name}")

        # valeurs par défaut + sim_start TÔT pour éviter 1970
        ensure_export_defaults(run_cfg)
        pipeline, ctx = build(run_cfg)
        ensure_sim_start_on_ctx(ctx, run_cfg)

        # Affichage param altitude (pour debug)
        alt = (run_cfg.get("altitude", {}) or run_cfg.get("plugins", {}).get("altitude", {}) or {})
        base_url = alt.get("base_url") or "http://localhost:5004"
        timeout = float(alt.get("timeout", 30.0))
        print(f"[ALT] Using base_url={base_url}, timeout={timeout:.1f}s")

        # Patch pipeline (Exporter core + FlexisExporter, Altitude, ordre)
        patch_pipeline_for_flexis(pipeline, run_cfg)

        # Run
        print("[RUN] Executing…")
        if hasattr(pipeline, "run"): _ = pipeline.run(ctx)
        else: _ = pipeline(ctx)  # type: ignore
        print("[RUN] Done.")

        # Post-run hooks (export flexis additionnel + rapport HTML)
        try: call_export_flexis_post(ctx, run_cfg)
        except Exception as e: print(f"[Flexis] export_flexis post-run failed: {e}")
        try: call_flexis_report_post(ctx, run_cfg)
        except Exception as e: print(f"[Flexis] flexis_report post-run failed: {e}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())