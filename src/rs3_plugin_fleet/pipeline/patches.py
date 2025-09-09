# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from rs3_plugin_fleet.utils.time_utils import (
    ensure_sim_start_on_ctx,
    ensure_relative_time_columns,
    trim_tail_stop,
)


def _stage_name(s) -> str:
    return getattr(s, "name", getattr(s, "__class__", type("X", (), {})).__name__)


def _ctx_df(ctx):
    df = getattr(ctx, "df", None)
    if df is None:
        df = getattr(ctx, "data", None)
    return df


# ---------- AltitudeStage injector -------------------------------------------
def ensure_altitude_stage(pipeline, run_cfg: Dict[str, Any]) -> None:
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


# ---------- Exporter shim -----------------------------------------------------
class _StageRunnerShim:
    """Adapte un exporter exposant process(df, ctx) à l'API pipeline run(ctx)."""
    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "Exporter")
    def run(self, ctx):
        df = _ctx_df(ctx)
        if df is None:
            df = pd.DataFrame()
        out = self.inner.process(df, ctx)
        try:
            setattr(ctx, "df", out)
        except Exception:
            pass
        return True


class FlexisExportFromEnricher:
    """
    Enrichit flexis_* + assure t_ms + altitude-fallback + coupe arrêt terminal,
    puis délègue à utils/flexis_export.Stage (sous forme de runner).
    """
    name = "FlexisExporter"

    def __init__(self, exporter_factory):
        self._make_exporter = exporter_factory

    def run(self, ctx):
        cfg_all = getattr(ctx, "config", {}) or {}
        ensure_sim_start_on_ctx(ctx, cfg_all)

        # enrichissement flexis_*
        try:
            from rs3_plugin_fleet.flexis import enricher as _flexis_enricher  # type: ignore
        except Exception:
            _flexis_enricher = None  # type: ignore

        try:
            if _flexis_enricher is not None:
                EnrCls = getattr(_flexis_enricher, "FlexisEnricher", None)
                if EnrCls is not None:
                    enr = EnrCls(cfg_all.get("flexis", {}) or {})
                    df = _ctx_df(ctx)
                    if df is not None:
                        new_df = enr.apply(df, context={"config": cfg_all})
                        try:
                            setattr(ctx, "df", new_df)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Flexis] Enricher apply failed (non-fatal): {e}")

        # garantir t_ms
        df2 = _ctx_df(ctx)
        df2 = ensure_relative_time_columns(df2, ctx)
        try:
            if "t_ms" in df2.columns:
                df2["t_ms"] = pd.to_numeric(df2["t_ms"], errors="coerce").fillna(0).astype("int64")
        except Exception:
            pass
        setattr(ctx, "df", df2)

        # altitude fallback: si aucun AltitudeStage n'a tourné, tente un enrichissement direct
        try:
            from rs3_plugin_fleet.plugin_discovery import altitude_loader as _alt
            df_alt = _ctx_df(ctx)
            if df_alt is not None:
                for fn_name in ("enrich_altitude", "add_altitude", "apply", "process", "enrich"):
                    fn = getattr(_alt, fn_name, None)
                    if callable(fn):
                        try:
                            new_alt_df = fn(df_alt, ctx) if fn.__code__.co_argcount >= 2 else fn(df_alt)
                            if isinstance(new_alt_df, pd.DataFrame) and "altitude" in new_alt_df.columns:
                                setattr(ctx, "df", new_alt_df)
                                print("[ADAPTER] Altitude applied via altitude_loader fallback")
                                break
                        except Exception:
                            continue
        except Exception:
            pass

        # couper un éventuel arrêt terminal artificiel
        try:
            trimmed = trim_tail_stop(_ctx_df(ctx), ctx, cfg_all)
            if isinstance(trimmed, pd.DataFrame) and len(trimmed) > 0:
                setattr(ctx, "df", trimmed)
        except Exception:
            pass

        # export Flexis
        exp = self._make_exporter()
        if not hasattr(exp, "run") and hasattr(exp, "process"):
            exp = _StageRunnerShim(exp)
        _ = exp.run(ctx)
        return True


# ---------- Patch pipeline ----------------------------------------------------
def patch_pipeline_for_flexis(pipeline, run_cfg: Dict[str, Any]) -> None:
    """
    - Inject AltitudeStage si dispo
    - Supprime FinalTailZero (pour enlever l'arrêt forcé)
    - Conserve Exporter core + ajoute FlexisExporter juste après → et remet FlexisExporter en TOUT DERNIER
    - Force SpeedSync avant IMUProjector & Exporter
    """
    stages = getattr(pipeline, "stages", None)
    if not isinstance(stages, list):
        return

    ensure_altitude_stage(pipeline, run_cfg)
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
        inst = FlexisExportFromEnricher(_exporter_factory)
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


# ---------- Post-run hooks ----------------------------------------------------
def call_flexis_report_post(ctx, cfg: Dict[str, Any]) -> None:
    try:
        from rs3_plugin_fleet.report import flexis_report as _fr
    except Exception:
        return
    df = _ctx_df(ctx)
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


def call_export_flexis_post(ctx, cfg: Dict[str, Any]) -> None:
    try:
        from rs3_plugin_fleet.flexis import enricher as _flexis_enricher  # type: ignore
    except Exception:
        return
    fn = getattr(_flexis_enricher, "export_flexis", None)
    if not callable(fn):
        return
    df = _ctx_df(ctx)
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