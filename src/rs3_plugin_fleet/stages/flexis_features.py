# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import pandas as pd

# Single source of truth: reuse FlexisEnricher feature logic
from rs3_plugin_fleet.flexis.enricher import FlexisEnricher


class FlexisFeaturesStage:
    """Pipeline stage wrapper that delegates to FlexisEnricher.
    Supports both core2 Context API (run(ctx)) and df-in/df-out usage.
    """

    name = "FlexisFeaturesStage"

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs):
        # Accept arbitrary kwargs coming from YAML/profile (e.g. hz, osrm, etc.)
        merged = {}
        if isinstance(config, dict):
            merged.update(config)
        if kwargs:
            merged.update(kwargs)

        # Only pass-through keys the enricher understands
        allowed = {
            "traffic_profile",
            "weather_timeline",
            "infra_probability_per_km",
            "driver_events_rate_per_hour",
            "population_density_mode",
        }
        filtered = {k: v for k, v in merged.items() if k in allowed}
        self._cfg = filtered
        self._enricher = FlexisEnricher(filtered)

    # --- core2 pipeline entrypoint ---
    def run(self, arg, *_, **__):
        """If called with a Context, mutate it in-place; otherwise treat `arg` as a DataFrame."""
        # Case 1: core2 Context object (common: has attribute `df`)
        if hasattr(arg, "df"):
            ctx = arg
            df_in = getattr(ctx, "df")
            # Best-effort extraction of a dict-like context
            ctx_dict = {}
            meta = getattr(ctx, "meta", None)
            if isinstance(meta, dict):
                ctx_dict.update(meta)
            extra = getattr(ctx, "context", None)
            if isinstance(extra, dict):
                ctx_dict.update(extra)
            # Enrich and write back
            df_out = self._enricher.apply(df_in, context=ctx_dict)
            setattr(ctx, "df", df_out)
            return ctx

        # Case 2: plain DataFrame usage
        if isinstance(arg, pd.DataFrame):
            return self._enricher.apply(arg, context={})

        # Fallback: return arg unchanged if we cannot handle the type
        return arg

    # Compatibility aliases used by some pipelines
    def process(self, df: pd.DataFrame, context: Dict[str, Any] | None = None) -> pd.DataFrame:
        return self._enricher.apply(df, context or {})

    def __call__(self, df: pd.DataFrame, context: Dict[str, Any] | None = None) -> pd.DataFrame:
        return self._enricher.apply(df, context or {})