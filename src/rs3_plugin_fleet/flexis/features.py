# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import pandas as pd

# On réutilise directement la logique de features de l'enricher
from rs3_plugin_fleet.flexis.enricher import FlexisEnricher

class FlexisFeaturesStage:
    """Wrapper de stage core2 qui délègue à FlexisEnricher.
       - Accepte un Context (run(ctx)) OU un DataFrame (appel direct).
       - Passe le contexte complet à l'enricher pour qu'il récupère
         traffic_profile / weather_timeline depuis ctx.config.profiles.<profil>.
    """
    name = "flexis"

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs):
        merged = {}
        if isinstance(config, dict):
            merged.update(config)
        if kwargs:
            merged.update(kwargs)

        # Seuls les paramètres compris par l'enricher
        allowed = {
            "traffic_profile",
            "weather_timeline",
            "infra_probability_per_km",
            "driver_events_rate_per_hour",
            "population_density_mode",
            "timezone",
        }
        filtered = {k: v for k, v in merged.items() if k in allowed}
        self._cfg = filtered
        self._enricher = FlexisEnricher(filtered)

    # --- core2 pipeline entrypoint ---
    def run(self, arg, *_, **__):
        # 1) Contexte core2
        if hasattr(arg, "df"):
            ctx = arg
            df_in = getattr(ctx, "df")
            # Passer le *contexte complet* pour extraire les profils du YAML
            df_out = self._enricher.apply(df_in, context=ctx)
            setattr(ctx, "df", df_out)
            return ctx

        # 2) DataFrame direct
        if isinstance(arg, pd.DataFrame):
            return self._enricher.apply(arg, context=None)

        # 3) fallback
        return arg

    # Compatibilité
    def process(self, df: pd.DataFrame, context: Dict[str, Any] | None = None) -> pd.DataFrame:
        return self._enricher.apply(df, context or {})

    def __call__(self, df: pd.DataFrame, context: Dict[str, Any] | None = None) -> pd.DataFrame:
        return self._enricher.apply(df, context or {})