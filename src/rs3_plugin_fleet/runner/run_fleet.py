# -*- coding: utf-8 -*-
"""
Runner Fleet — lance une simulation multi-profils à partir d'un YAML.

Usage:
  python -m rs3_plugin_fleet.runner.run_fleet --config path/to/coin-coin-delivery.yaml

Fonctions :
- Charge la config YAML (avec support d'un champ 'profiles'/'fleet').
- Normalise `ctx.sim_start` (UTC) tôt pour éviter les timestamps 1970 plus loin.
- Construit le pipeline via un "builder" ou un "adapter" dynamique, selon ce qui est dispo.
- Affiche les stages découverts et quelques infos utiles (Altitude service, etc.).
- Exécute le pipeline et remplace l'Exporter core par un exporteur Flexis orchestré via enricher.py.

Noms d’imports :
- Essaie d'abord rs3_plugin_fleet.adapters.core2_adapter_dyn puis rs3_plugin_fleet.pipeline.builder
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Dict

import pandas as pd
import yaml

# ---- Embedded Flexis exporter fallback (used if flexis_export.Stage missing) ---
import os as _os
import json as _json
from dataclasses import dataclass as _dataclass
import pandas as _pd

def _fe_infer_sim_start_from_config(cfg: Dict[str, Any]) -> _pd.Timestamp:
    start_at = (
        cfg.get("start_at")
        or (cfg.get("simulation", {}) or {}).get("start_at")
        or (cfg.get("fleet", {}) or {}).get("start_at")
    )
    if start_at:
        ts = _pd.Timestamp(start_at)
        if ts.tz is None:
            ts = ts.tz_localize("Europe/Paris").tz_convert("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    return _pd.Timestamp.now(tz="UTC")

@_dataclass
class FlexisExporter:
    """Minimal Flexis-aware exporter (fallback)."""
    name: str = "Exporter"

    def process(self, df: _pd.DataFrame, ctx: Any) -> _pd.DataFrame:
        cfg = getattr(ctx, "config", {}) or {}
        out_dir = (
            (cfg.get("output", {}) or {}).get("dir")
            or (cfg.get("export", {}) or {}).get("dir")
            or "data/simulations/default"
        )
        _os.makedirs(out_dir, exist_ok=True)

        export_cfg = (cfg.get("export", {}) or {})
        fmt = (export_cfg.get("format") or "parquet").lower()
        filename = export_cfg.get("filename") or ("flexis.parquet" if fmt == "parquet" else "flexis.csv")
        events_filename = export_cfg.get("events_filename") or "events.jsonl"

        sim_start = getattr(ctx, "sim_start", None)
        if sim_start is None:
            sim_start = _fe_infer_sim_start_from_config(cfg)
            try: setattr(ctx, "sim_start", sim_start)
            except Exception: pass

        # Compute absolute time from t_ms/t_s if available
        out = df.copy()
        if "t_ms" in out:
            abs_time = (_pd.to_datetime(sim_start) + _pd.to_timedelta(out["t_ms"], unit="ms")).dt.tz_convert("UTC")
        elif "t_s" in out:
            abs_time = (_pd.to_datetime(sim_start) + _pd.to_timedelta(out["t_s"], unit="s")).dt.tz_convert("UTC")
        else:
            # If no relative time, try to use an existing datetime column
            if "timestamp" in out.columns:
                abs_time = _pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            elif "time_utc" in out.columns:
                abs_time = _pd.to_datetime(out["time_utc"], utc=True, errors="coerce")
            else:
                abs_time = _pd.to_datetime([], utc=True, errors="coerce")
        out["time_utc"] = abs_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out["ts_ms"] = (abs_time.astype("int64") // 1_000_000).astype("int64")

        # Tail-stop safety (optional)
        if bool(export_cfg.get("close_tail_stop", True)) and len(out) > 0 and "in_stop" in out.columns:
            tail_window = int(export_cfg.get("tail_window", 5))
            tail = out.tail(tail_window)
            all_stopped = True
            if "speed_mps" in out.columns:
                all_stopped = (tail["speed_mps"].abs() < 0.05).all()
            if all_stopped and tail["in_stop"].all():
                last_idx = out.index[-1]
                if len(out) >= 2:
                    prev_idx = out.index[-2]
                    out.at[prev_idx, "in_stop"] = False
                out.at[last_idx, "in_stop"] = False
                if "in_delivery" in out.columns:
                    out.at[last_idx, "in_delivery"] = False
                if "delivery_state" in out.columns:
                    out.at[last_idx, "delivery_state"] = "done"
                out.loc[last_idx, "run_state"] = "finished"

        fpath = _os.path.join(out_dir, filename)
        if fmt == "csv":
            out.to_csv(fpath, index=False)
        elif fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
                out.to_parquet(fpath, index=False)
            except Exception:
                fallback = _os.path.splitext(fpath)[0] + ".csv"
                out.to_csv(fallback, index=False)
                fpath = fallback
        else:
            fallback = _os.path.splitext(fpath)[0] + ".csv"
            out.to_csv(fallback, index=False)
            fpath = fallback

        if hasattr(ctx, "logger"):
            ctx.logger.info(f"[Exporter] Wrote {fpath}")
        else:
            print(f"[Exporter] Wrote {fpath}")

        # Events (optional)
        events = getattr(ctx, "events", None)
        if isinstance(events, (list, tuple)) and len(events) > 0:
            epath = _os.path.join(out_dir, events_filename)
            with open(epath, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(_json.dumps(ev, ensure_ascii=False) + "\n")
            if hasattr(ctx, "logger"):
                ctx.logger.info(f"[Exporter] Wrote {epath}")
            else:
                print(f"[Exporter] Wrote {epath}")

        return out

    # Pipeline compatibility
    def run(self, ctx: Any) -> _pd.DataFrame:
        df = getattr(ctx, "df", None)
        if df is None:
            df = getattr(ctx, "data", None)
        if df is None:
            df = _pd.DataFrame()
        out = self.process(df, ctx)
        try:
            setattr(ctx, "df", out)
        except Exception:
            pass
        # Return a simple truthy value instead of a DataFrame (avoids pandas truth-value ambiguity)
        return True

# ---- Flexis enricher integration -------------------------------------------
try:
    # Chemin recommandé : rs3_plugin_fleet/flexis/enricher.py
    from rs3_plugin_fleet.flexis import enricher as _flexis_enricher  # type: ignore
except Exception:
    _flexis_enricher = None  # type: ignore

# --------- Utilities ---------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _tz_aware_utc_from_config(cfg: Dict[str, Any]) -> pd.Timestamp:
    start_at = (
        cfg.get("start_at")
        or cfg.get("simulation", {}).get("start_at")
        or cfg.get("fleet", {}).get("start_at")
    )
    if start_at:
        ts = pd.Timestamp(start_at)
        if ts.tz is None:
            ts = ts.tz_localize("Europe/Paris").tz_convert("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    return pd.Timestamp.now(tz="UTC")

def _print_pipeline(stages) -> None:
    names = [getattr(s, "name", s.__class__.__name__) for s in stages]
    chain = " → ".join(names)
    print(f"[PIPELINE] Stages before final clamp: {chain}")

def _stage_name(s) -> str:
    return getattr(s, "name", getattr(s, "__class__", type("X", (), {})).__name__)

def _ensure_export_defaults(run_cfg: Dict[str, Any]) -> None:
    """
    Garantit output/export/start_at pour éviter 1970 et les chemins manquants.
    Mutates run_cfg in-place.
    """
    # output.dir
    out_dir = (
        run_cfg.get("output", {}).get("dir")
        or run_cfg.get("export", {}).get("dir")
        or run_cfg.get("outdir")
        or "data/simulations/default"
    )
    run_cfg.setdefault("output", {})
    run_cfg["output"]["dir"] = out_dir

    # export block
    exp = run_cfg.setdefault("export", {})
    exp.setdefault("format", "parquet")
    exp.setdefault("filename", "flexis.parquet")
    exp.setdefault("events_filename", "events.jsonl")
    exp.setdefault("close_tail_stop", True)
    exp.setdefault("tail_window", 5)

    # start_at: défaut = aujourd'hui 08:00 Europe/Paris (runner convertit en UTC)
    if "start_at" not in run_cfg:
        ts_local = pd.Timestamp(pd.Timestamp.now(tz="Europe/Paris").date())
        run_cfg["start_at"] = ts_local.replace(hour=8, minute=0, second=0).strftime("%Y-%m-%d %H:%M")

class _StageRunnerShim:
    """
    Adapte un exporter exposant process(df, ctx) à l'API pipeline run(ctx).
    """
    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "Exporter")
    def run(self, ctx):
        df = getattr(ctx, "df", None)
        if df is None:
            df = getattr(ctx, "data", None)
        if df is None:
            import pandas as _pd
            df = _pd.DataFrame()
        out = self.inner.process(df, ctx)
        try:
            setattr(ctx, "df", out)
        except Exception:
            pass
        # Return a simple truthy value instead of a DataFrame (avoids pandas truth-value ambiguity)
        return True

class _FlexisExportFromEnricher:
    """
    Stage composite : applique FlexisEnricher (si dispo), puis délègue
    l'écriture au véritable exporteur Flexis.
    """
    name = "FlexisExporter"
    def __init__(self, exporter_factory):
        # exporter_factory() -> instance avec run(ctx) ou process(df, ctx)
        self._make_exporter = exporter_factory
    def run(self, ctx):
        # 1) enrichissement flexis_* (optionnel)
        try:
            if _flexis_enricher is not None:
                EnrCls = getattr(_flexis_enricher, "FlexisEnricher", None)
                if EnrCls is not None:
                    cfg_all = getattr(ctx, "config", {}) or {}
                    enr_cfg = cfg_all.get("flexis", {}) or {}
                    enr = EnrCls(enr_cfg)
                    df = getattr(ctx, "df", None)
                    if df is None:
                        df = getattr(ctx, "data", None)
                    if df is not None:
                        new_df = enr.apply(df, context={"config": cfg_all})
                        try:
                            setattr(ctx, "df", new_df)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Flexis] Enricher apply failed (non-fatal): {e}")
        # 2) export
        exp = self._make_exporter()
        if not hasattr(exp, "run") and hasattr(exp, "process"):
            exp = _StageRunnerShim(exp)
        _ = exp.run(ctx)
        return True

def _patch_pipeline_for_flexis(pipeline, run_cfg: Dict[str, Any]) -> None:
    """
    - Supprime FinalTailZero (arrêt prolongé).
    - Remplace le core Exporter par un exporteur Flexis : enricher → exporter.
    - Impose l'ordre : SpeedSync avant IMUProjector et avant Exporter ; Exporter en dernier.
    """
    stages = getattr(pipeline, "stages", None)
    if not isinstance(stages, list):
        return

    # 1) remove FinalTailZero
    stages[:] = [s for s in stages if _stage_name(s) != "FinalTailZero"]

    # 2) fabrique d'exporteur Flexis (préférence au module flexis_export)
    def _exporter_factory():
        # a) Try the project-provided exporter
        try:
            from rs3_plugin_fleet.flexis_export import Stage as _FlexisExporterStage  # type: ignore
            return _FlexisExporterStage()
        except Exception:
            pass
        # b) Fallback to the embedded exporter defined above
        return FlexisExporter()

    # 3) ajouter un stage FlexisExporter (en plus du core Exporter)
    exp_idx = next((i for i, s in enumerate(stages) if _stage_name(s) == "Exporter"), None)
    has_flexis_exp = any(_stage_name(s) == "FlexisExporter" for s in stages)
    try:
        if not has_flexis_exp:
            inst = _FlexisExportFromEnricher(_exporter_factory)
            if exp_idx is not None:
                insert_pos = exp_idx + 1
                stages.insert(insert_pos, inst)
                print("[ADAPTER] Inserted FlexisExporter after core Exporter")
            else:
                stages.append(inst)
                print("[ADAPTER] Appended FlexisExporter at pipeline end (no core Exporter found)")
    except Exception as e:
        print(f"[ADAPTER] Warning: could not construct Flexis enricher-backed exporter: {e}")

    # 4) imposer l'ordre : SpeedSync avant IMUProjector et Exporter
    def _move_before(name_a, name_b):
        a = next((i for i, s in enumerate(stages) if _stage_name(s) == name_a), None)
        b = next((i for i, s in enumerate(stages) if _stage_name(s) == name_b), None)
        if a is not None and b is not None and a > b:
            item = stages.pop(a)
            b = next((i for i, s in enumerate(stages) if _stage_name(s) == name_b), None)
            stages.insert(b, item)
    _move_before("SpeedSync", "IMUProjector")
    _move_before("SpeedSync", "Exporter")

    def _move_after(name_a, name_b):
        # move stage name_a to the position immediately after name_b
        a = next((i for i, s in enumerate(stages) if _stage_name(s) == name_a), None)
        b = next((i for i, s in enumerate(stages) if _stage_name(s) == name_b), None)
        if a is not None and b is not None and a <= b:
            item = stages.pop(a)
            # recompute b after pop
            b = next((i for i, s in enumerate(stages) if _stage_name(s) == name_b), None)
            stages.insert(b + 1, item)

    # garantir l'ordre: core Exporter puis FlexisExporter
    _move_after("FlexisExporter", "Exporter")

    # 5) FlexisExporter strictement dernier (le core Exporter reste juste avant)
    flex_idx = next((i for i, s in enumerate(stages) if _stage_name(s) == "FlexisExporter"), None)
    if flex_idx is not None and flex_idx != len(stages) - 1:
        item = stages.pop(flex_idx)
        stages.append(item)

def _call_export_flexis_post(ctx, cfg: Dict[str, Any]) -> None:
    """
    Filet post-run : si enricher expose `export_flexis(...)`, on l'appelle.
    Signatures testées : (df, ctx, cfg) → (df, cfg) → (df) → ().
    """
    if _flexis_enricher is None:
        return
    fn = getattr(_flexis_enricher, "export_flexis", None)
    if not callable(fn):
        return
    df = getattr(ctx, "df", None)
    if df is None:
        df = getattr(ctx, "data", None)
    try:
        fn(df, ctx, cfg)
    except TypeError:
        try:
            fn(df, cfg)
        except TypeError:
            try:
                fn(df)
            except TypeError:
                fn()
    print("[Flexis] export_flexis() invoked post-run")

# --------- Try imports for builder/adapter -----------------------------------

def _get_builder():
    """
    Renvoie un callable (cfg) -> (pipeline, ctx).
    Ordre d'essai et comportements:
      1) rs3_plugin_fleet.adapters.core2_adapter_dyn
         - si `build_pipeline_and_context` existe, on l'utilise
         - sinon, on construit un builder à partir de get_pipeline_cls/get_context_cls/build_default_stages
      2) rs3_plugin_fleet.pipeline.builder
         - si `build_pipeline_and_context` existe, on l'utilise
    """
    # 1) Adapter dynamique (chemin paquet)
    try:
        import rs3_plugin_fleet.adapters.core2_adapter_dyn as adapter
        if hasattr(adapter, "build_pipeline_and_context"):
            return adapter.build_pipeline_and_context  # type: ignore
        # Fallback: construire un builder à partir des getters exposés
        if all(hasattr(adapter, n) for n in ("get_pipeline_cls", "get_context_cls", "build_default_stages")):
            def _from_adapter(cfg):
                PipelineCls = adapter.get_pipeline_cls()
                ContextCls = adapter.get_context_cls()
                stages = adapter.build_default_stages(cfg)
                pipeline = PipelineCls(stages)
                try:
                    ctx = ContextCls(cfg)  # some Context implementations require cfg
                except TypeError:
                    ctx = ContextCls()
                try:
                    setattr(ctx, "config", cfg)
                except Exception:
                    pass
                return pipeline, ctx
            return _from_adapter
    except Exception:
        pass

    # 2) Adapter dynamique (chemin relatif)
    try:
        from ..adapters import core2_adapter_dyn as adapter  # type: ignore
        if hasattr(adapter, "build_pipeline_and_context"):
            return adapter.build_pipeline_and_context  # type: ignore
        if all(hasattr(adapter, n) for n in ("get_pipeline_cls", "get_context_cls", "build_default_stages")):
            def _from_adapter_rel(cfg):
                PipelineCls = adapter.get_pipeline_cls()
                ContextCls = adapter.get_context_cls()
                stages = adapter.build_default_stages(cfg)
                pipeline = PipelineCls(stages)
                try:
                    ctx = ContextCls(cfg)  # some Context implementations require cfg
                except TypeError:
                    ctx = ContextCls()
                try:
                    setattr(ctx, "config", cfg)
                except Exception:
                    pass
                return pipeline, ctx
            return _from_adapter_rel
    except Exception:
        pass

    # 3) Builder classique (chemin paquet)
    try:
        from rs3_plugin_fleet.pipeline.builder import build_pipeline_and_context  # type: ignore
        return build_pipeline_and_context
    except Exception:
        pass

    # 4) Builder classique (chemin relatif)
    try:
        from ..pipeline.builder import build_pipeline_and_context  # type: ignore
        return build_pipeline_and_context
    except Exception:
        pass

    raise ImportError(
        "Impossible d'importer/constituer build_pipeline_and_context depuis adapters/core2_adapter_dyn ou pipeline/builder."
    )

# --------- main --------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RS3 Fleet Runner")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Chemin vers le YAML de configuration (ex: coin-coin-delivery.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    profile_name = cfg.get("name") or cfg.get("profile") or cfg.get("fleet", {}).get("name")
    profile_info = cfg.get("profile") or cfg.get("fleet", {}).get("profile")
    if profile_name:
        if profile_info:
            print(f"[RUN] {profile_name} ← profil={profile_info}")
        else:
            print(f"[RUN] {profile_name}")

    # ---------- Build helper ----------
    build = _get_builder()

    def _expand_vehicle_run_cfg(cfg_root: Dict[str, Any], vehicle: Dict[str, Any]) -> Dict[str, Any]:
        """Compose a per-vehicle run config expected by the core pipeline.
        Merges the selected profile into a flat dict and injects the vehicle's stops.
        """
        run: Dict[str, Any] = {}
        profiles = cfg_root.get("profiles", {}) or {}
        v_profile = vehicle.get("profile")
        prof = profiles.get(v_profile, {}) if v_profile else {}
        # start with profile block
        run.update(deepcopy(prof))
        # carry top-level knobs if present (do not override explicit profile keys)
        for k in ("osrm","speed_sync","road_enrich","geo_spike_filter","legs_retimer","stop_wait_injector","stop_smoother","validators","exporter","altitude","flexis","stages"):
            if k in cfg_root and k not in run:
                run[k] = deepcopy(cfg_root[k])
        # inject vehicle info
        run["vehicle_id"] = vehicle.get("id")
        run["stops"] = deepcopy(vehicle.get("stops", []))
        # name/profile for logs
        run["name"] = cfg_root.get("client") or cfg_root.get("name") or run.get("name") or run.get("vehicle_id")
        run["profile"] = v_profile or run.get("profile")
        # start_at inherit if not present
        if "start_at" not in run and "start_at" in cfg_root:
            run["start_at"] = cfg_root["start_at"]
        # output.dir inherit
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

    # If no vehicles defined, assume cfg already describes a single-run config
    runs = []
    if vehicles:
        for v in vehicles:
            runs.append(_expand_vehicle_run_cfg(cfg, v))
    else:
        runs.append(cfg)

    exit_code = 0

    for i, run_cfg in enumerate(runs, start=1):
        name = run_cfg.get("vehicle_id") or run_cfg.get("name") or f"run-{i:02d}"
        prof = run_cfg.get("profile")
        if name and prof:
            print(f"[RUN] {name} ← profil={prof}")
        elif name:
            print(f"[RUN] {name}")

        # Construit pipeline + contexte
        _ensure_export_defaults(run_cfg)
        pipeline, ctx = build(run_cfg)

        # Normalise sim_start tôt
        if not hasattr(ctx, "sim_start"):
            ctx.sim_start = _tz_aware_utc_from_config(run_cfg)

        # Petites infos : altitude service (affiché pour info)
        alt = (run_cfg.get("altitude", {}) or run_cfg.get("plugins", {}).get("altitude", {}) or {})
        base_url = alt.get("base_url") or "http://localhost:5004"
        timeout = float(alt.get("timeout", 30.0))
        print(f"[ALT] Using base_url={base_url}, timeout={timeout:.1f}s")

        try:
            stages = getattr(pipeline, "stages", None) or []
            _print_pipeline(stages)
        except Exception:
            pass

        # Patch pipeline to use Flexis enricher-backed Exporter
        _patch_pipeline_for_flexis(pipeline, run_cfg)
        try:
            stages = getattr(pipeline, "stages", None) or []
            print("[PIPELINE] Stages after flexis patch:")
            _print_pipeline(stages)
        except Exception:
            pass

        # Exécution — on autorise deux modes : pipeline.run(ctx) ou pipeline(ctx)
        print("[RUN] Executing…")
        if hasattr(pipeline, "run"):
            _ = pipeline.run(ctx)
        else:
            _ = pipeline(ctx)  # type: ignore
        print("[RUN] Done.")

        # Post-run safety net: export_flexis(...) si présent
        try:
            _call_export_flexis_post(ctx, run_cfg)
        except Exception as e:
            print(f"[Flexis] export_flexis post-run failed: {e}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
