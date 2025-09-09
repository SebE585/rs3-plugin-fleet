# -*- coding: utf-8 -*-
"""
Runner Fleet — lance une simulation multi-profils à partir d'un YAML.

- Normalise ctx.sim_start (UTC) TRÈS tôt → plus de dates 1970.
- Injecte AltitudeStage si dispo.
- Maintient Exporter core + ajoute FlexisExporter (dernier).
- Filets post-run : export_flexis() (si présent) + rapport Flexis HTML.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Dict

import pandas as pd
import yaml

# ---------- Robust UTC timestamp parser --------------------------------------
def _parse_ts_utc(ts: pd.Series) -> pd.Series:
    """Parse une série hétérogène en datetimes UTC (epoch s/ms/ns, datetimes, strings)."""
    if ts is None:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    s = pd.to_numeric(ts, errors="coerce")
    if s.notna().any():
        mx = s.dropna().abs().max()
        try:
            if mx < 1e11:      # secondes
                return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            elif mx < 1e14:    # millisecondes
                return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            else:              # nanosecondes
                return pd.to_datetime(s, unit="ns", utc=True, errors="coerce")
        except Exception:
            pass
    return pd.to_datetime(ts, utc=True, errors="coerce")

# ---------- Flexis enricher (optionnel) --------------------------------------
try:
    from rs3_plugin_fleet.flexis import enricher as _flexis_enricher  # type: ignore
except Exception:
    _flexis_enricher = None  # type: ignore

# ---------- Utils ------------------------------------------------------------
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
    print(f"[PIPELINE] Stages before final clamp: {' → '.join(names)}")

def _stage_name(s) -> str:
    return getattr(s, "name", getattr(s, "__class__", type("X", (), {})).__name__)

def _ensure_export_defaults(run_cfg: Dict[str, Any]) -> None:
    """Garantit output/export/start_at pour éviter 1970 et assurer les sorties."""
    out_dir = (
        run_cfg.get("output", {}).get("dir")
        or run_cfg.get("export", {}).get("dir")
        or run_cfg.get("outdir")
        or "data/simulations/default"
    )
    run_cfg.setdefault("output", {})
    run_cfg["output"]["dir"] = out_dir
    exp = run_cfg.setdefault("export", {})
    exp.setdefault("format", "parquet")
    exp.setdefault("filename", "flexis.parquet")
    exp.setdefault("events_filename", "events.jsonl")
    exp.setdefault("close_tail_stop", True)
    exp.setdefault("tail_window", 5)  # secondes
    if "start_at" not in run_cfg:
        ts_local = pd.Timestamp(pd.Timestamp.now(tz="Europe/Paris").date())
        run_cfg["start_at"] = ts_local.replace(hour=8, minute=0, second=0).strftime("%Y-%m-%d %H:%M")

def _ensure_sim_start_on_ctx(ctx: Any, cfg: Dict[str, Any]) -> pd.Timestamp:
    """Consolide ctx.sim_start → tz-aware UTC (utilisé par core et export Flexis)."""
    sim_start = getattr(ctx, "sim_start", None)
    if sim_start is None:
        sim_start = _tz_aware_utc_from_config(cfg)
        try:
            setattr(ctx, "sim_start", sim_start)
        except Exception:
            pass
    else:
        ts = pd.Timestamp(sim_start)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        setattr(ctx, "sim_start", ts)
        sim_start = ts
    return sim_start

def _ensure_relative_time_columns(df: pd.DataFrame, ctx: Any) -> pd.DataFrame:
    """
    Reconstruit t_ms/t_s si absents à partir de timestamp/time_utc/ts_ms + ctx.sim_start.
    Clamp ±100 ans pour éviter OutOfBoundsTimedelta.
    """
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    if ("t_ms" in df.columns) or ("t_s" in df.columns):
        return df

    sim_start = getattr(ctx, "sim_start", None) or pd.Timestamp.now(tz="UTC")
    sim_start = pd.Timestamp(sim_start).tz_convert("UTC")
    start_ms = int(sim_start.value // 1_000_000)

    # ts_ms absolu → relatif
    if "ts_ms" in df.columns:
        ts_raw = pd.to_numeric(df["ts_ms"], errors="coerce")
        if ts_raw.notna().any():
            mx = ts_raw.dropna().abs().max()
            if mx < 1e11:   # s
                ts_ms = ts_raw * 1000.0
            elif mx < 1e14: # ms
                ts_ms = ts_raw
            else:           # ns
                ts_ms = ts_raw / 1e6
            t_ms = ts_ms - start_ms
            clip_win = int(100 * 365.25 * 24 * 3600 * 1000)
            df2 = df.copy()
            df2["t_ms"] = pd.to_numeric(t_ms, errors="coerce").fillna(0).clip(-clip_win, clip_win).astype("int64")
            return df2

    # sinon timestamp/time_utc
    ts_abs = None
    if "timestamp" in df.columns:
        ts_abs = _parse_ts_utc(df["timestamp"])
    elif "time_utc" in df.columns:
        ts_abs = _parse_ts_utc(df["time_utc"])
    if ts_abs is not None:
        valid = ts_abs.notna()
        ts_ms = pd.Series(0, index=df.index, dtype="int64")
        if valid.any():
            ts_ms.loc[valid] = (ts_abs[valid].astype("int64") // 1_000_000).astype("int64")
        delta_ms = ts_ms - start_ms
        clip_win = int(100 * 365.25 * 24 * 3600 * 1000)
        df2 = df.copy()
        df2["t_ms"] = pd.to_numeric(delta_ms, errors="coerce").fillna(0).clip(-clip_win, clip_win).astype("int64")
        return df2

    return df

# ---------- AltitudeStage injector -------------------------------------------
def _ensure_altitude_stage(pipeline, run_cfg: Dict[str, Any]) -> None:
    stages = getattr(pipeline, "stages", None)
    if not isinstance(stages, list):
        return
    if any(_stage_name(s) == "AltitudeStage" for s in stages):
        return
    try:
        from rs3_plugin_fleet.plugin_discovery import altitude_loader as _alt
    except Exception:
        return

    stage = None
    # AltitudeStage(config) si dispo…
    StageCls = getattr(_alt, "AltitudeStage", None)
    if StageCls:
        try:
            stage = StageCls(run_cfg.get("altitude", {}) or run_cfg.get("plugins", {}).get("altitude", {}) or {})
        except Exception:
            stage = None
    # …sinon fabrique
    if stage is None:
        for factory_name in ("build_stage", "build_altitude_stage", "get_stage"):
            fn = getattr(_alt, factory_name, None)
            if callable(fn):
                try:
                    stage = fn(run_cfg)
                    break
                except Exception:
                    stage = None
    if stage is None:
        return

    def _find(name):
        return next((i for i, s in enumerate(stages) if _stage_name(s) == name), None)
    anchor = _find("NoiseInjector") or _find("RoadEnricher")
    insert_pos = (anchor + 1) if anchor is not None else max(len(stages) - 1, 0)
    stages.insert(insert_pos, stage)
    print("[ADAPTER] Inserted AltitudeStage")

class _StageRunnerShim:
    """Adapte un exporter exposant process(df, ctx) à l'API pipeline run(ctx)."""
    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "Exporter")
    def run(self, ctx):
        df = getattr(ctx, "df", None) or getattr(ctx, "data", None) or pd.DataFrame()
        out = self.inner.process(df, ctx)
        try:
            setattr(ctx, "df", out)
        except Exception:
            pass
        return True

class _FlexisExportFromEnricher:
    """Enrichit flexis_* + assure t_ms puis délègue à utils/flexis_export.Stage."""
    name = "FlexisExporter"
    def __init__(self, exporter_factory):
        self._make_exporter = exporter_factory
    def run(self, ctx):
        cfg_all = getattr(ctx, "config", {}) or {}
        _ensure_sim_start_on_ctx(ctx, cfg_all)

        # enrichissement flexis_*
        try:
            if _flexis_enricher is not None:
                EnrCls = getattr(_flexis_enricher, "FlexisEnricher", None)
                if EnrCls is not None:
                    enr = EnrCls(cfg_all.get("flexis", {}) or {})
                    df = getattr(ctx, "df", None) or getattr(ctx, "data", None)
                    if df is not None:
                        new_df = enr.apply(df, context={"config": cfg_all})
                        setattr(ctx, "df", new_df)
        except Exception as e:
            print(f"[Flexis] Enricher apply failed (non-fatal): {e}")

        # garantir t_ms
        df2 = getattr(ctx, "df", None) or getattr(ctx, "data", None)
        df2 = _ensure_relative_time_columns(df2, ctx)
        try:
            if "t_ms" in df2.columns:
                df2["t_ms"] = pd.to_numeric(df2["t_ms"], errors="coerce").fillna(0).astype("int64")
        except Exception:
            pass
        setattr(ctx, "df", df2)

        # export Flexis
        exp = self._make_exporter()
        if not hasattr(exp, "run") and hasattr(exp, "process"):
            exp = _StageRunnerShim(exp)
        _ = exp.run(ctx)
        return True

def _patch_pipeline_for_flexis(pipeline, run_cfg: Dict[str, Any]) -> None:
    """
    - Inject AltitudeStage si dispo
    - Supprime FinalTailZero (pour enlever l'arrêt forcé)
    - Conserve Exporter core + ajoute FlexisExporter juste après → et remet FlexisExporter en TOUT DERNIER
    - Force SpeedSync avant IMUProjector & Exporter
    """
    stages = getattr(pipeline, "stages", None)
    if not isinstance(stages, list):
        return

    _ensure_altitude_stage(pipeline, run_cfg)
    stages[:] = [s for s in stages if _stage_name(s) != "FinalTailZero"]

    def _exporter_factory():
        try:
            from rs3_plugin_fleet.utils.flexis_export import Stage as _FlexisExporterStage  # type: ignore
            return _FlexisExporterStage()
        except Exception:
            class _Fallback:
                name = "Exporter"
                def process(self, df, ctx): return df
                def run(self, ctx): return True
            return _Fallback()

    exp_idx = next((i for i, s in enumerate(stages) if _stage_name(s) == "Exporter"), None)
    if not any(_stage_name(s) == "FlexisExporter" for s in stages):
        inst = _FlexisExportFromEnricher(_exporter_factory)
        if exp_idx is not None:
            stages.insert(exp_idx + 1, inst)
            print("[ADAPTER] Inserted FlexisExporter after core Exporter")
        else:
            stages.append(inst)
            print("[ADAPTER] Appended FlexisExporter at pipeline end (no core Exporter found)")

    def _move_before(name_a, name_b):
        a = next((i for i, s in enumerate(stages) if _stage_name(s) == name_a), None)
        b = next((i for i, s in enumerate(stages) if _stage_name(s) == name_b), None)
        if a is not None and b is not None and a > b:
            item = stages.pop(a)
            b2 = next((i for i, s in enumerate(stages) if _stage_name(s) == name_b), None)
            stages.insert(b2, item)
    _move_before("SpeedSync", "IMUProjector")
    _move_before("SpeedSync", "Exporter")

    flex_idx = next((i for i, s in enumerate(stages) if _stage_name(s) == "FlexisExporter"), None)
    if flex_idx is not None and flex_idx != len(stages) - 1:
        stages.append(stages.pop(flex_idx))

# ---------- Flexis HTML report post-run --------------------------------------
def _call_flexis_report_post(ctx, cfg: Dict[str, Any]) -> None:
    try:
        from rs3_plugin_fleet.report import flexis_report as _fr
    except Exception:
        return
    df = getattr(ctx, "df", None) or getattr(ctx, "data", None)
    for fn_name in ("generate_report", "write_report", "build_report", "generate_html"):
        fn = getattr(_fr, fn_name, None)
        if callable(fn):
            try:
                fn(df, cfg); print("[Report] Flexis report generated via", fn_name); return
            except TypeError:
                try:
                    fn(df); print("[Report] Flexis report generated via", fn_name); return
                except Exception:
                    continue
    try:
        ReportCls = getattr(_fr, "FlexisReport", None)
        if ReportCls is not None:
            rep = ReportCls(cfg)
            if hasattr(rep, "run"): rep.run(df, cfg)
            elif hasattr(rep, "process"): rep.process(df, cfg)
            print("[Report] Flexis report generated via FlexisReport class")
    except Exception:
        return

def _call_export_flexis_post(ctx, cfg: Dict[str, Any]) -> None:
    if _flexis_enricher is None:
        return
    fn = getattr(_flexis_enricher, "export_flexis", None)
    if not callable(fn):
        return
    df = getattr(ctx, "df", None) or getattr(ctx, "data", None)
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
        for k in ("osrm","speed_sync","road_enrich","geo_spike_filter","legs_retimer","stop_wait_injector","stop_smoother","validators","exporter","altitude","flexis","stages"):
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

        _ensure_export_defaults(run_cfg)
        pipeline, ctx = build(run_cfg)

        # sim_start LE PLUS TÔT POSSIBLE
        _ensure_sim_start_on_ctx(ctx, run_cfg)

        # Info altitude (affichage)
        alt = (run_cfg.get("altitude", {}) or run_cfg.get("plugins", {}).get("altitude", {}) or {})
        base_url = alt.get("base_url") or "http://localhost:5004"
        timeout = float(alt.get("timeout", 30.0))
        print(f"[ALT] Using base_url={base_url}, timeout={timeout:.1f}s")

        try:
            stages = getattr(pipeline, "stages", None) or []
            _print_pipeline(stages)
        except Exception:
            pass

        # Patch pipeline (Exporter core + FlexisExporter, Altitude, ordre)
        _patch_pipeline_for_flexis(pipeline, run_cfg)
        try:
            stages = getattr(pipeline, "stages", None) or []
            print("[PIPELINE] Stages after flexis patch:")
            _print_pipeline(stages)
        except Exception:
            pass

        # Run
        print("[RUN] Executing…")
        if hasattr(pipeline, "run"): _ = pipeline.run(ctx)
        else: _ = pipeline(ctx)  # type: ignore
        print("[RUN] Done.")

        # post-run filets
        try: _call_export_flexis_post(ctx, run_cfg)
        except Exception as e: print(f"[Flexis] export_flexis post-run failed: {e}")
        try: _call_flexis_report_post(ctx, run_cfg)
        except Exception as e: print(f"[Flexis] flexis_report post-run failed: {e}")

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())