# -*- coding: utf-8 -*-
"""
Flexis Exporter — écrit les résultats de simulation au format analytique (CSV/Parquet)
en garantissant des timestamps absolus corrects (UTC) et un état terminal propre.

Points clés :
- Convertit le temps relatif (t_ms ou t_s) → temps absolu UTC en s'appuyant sur ctx.sim_start.
- Si ctx.sim_start est absent, le déduit depuis config["start_at"] ou prend "now(UTC)".
- Ajoute deux colonnes : `ts_ms` (int64) et `time_utc` (ISO 8601, Z).
- Option "close_tail_stop": clôture un arrêt persistant en toute fin (filet de sécurité).
- Écrit DataFrame principal + éventuelle table d'événements dans export_dir.

Config attendue (exemples) :
  output:
    dir: "data/simulations/ccd_parcels_rouen"
  export:
    format: "parquet"        # "csv" ou "parquet" (par défaut parquet si pyarrow dispo)
    filename: "flexis.parquet"
    events_filename: "events.jsonl"
    close_tail_stop: true
    tail_window: 5
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _infer_sim_start_from_config(cfg: Dict[str, Any]) -> pd.Timestamp:
    """Retourne un pd.Timestamp tz-aware (UTC) à partir de cfg['start_at'] si présent;
    sinon retourne now(UTC). Si start_at est naïf, on assume Europe/Paris puis conversion UTC.
    """
    start_at = None
    # supporte plusieurs couches possibles
    if isinstance(cfg, dict):
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
    return pd.Timestamp.utcnow().tz_localize("UTC")


def _compute_abs_time(df: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
    """Calcule la série de timestamps absolus (tz-aware UTC) à partir de t_ms ou t_s."""
    if "t_ms" in df:
        dt = pd.to_timedelta(df["t_ms"], unit="ms")
    elif "t_s" in df:
        dt = pd.to_timedelta(df["t_s"], unit="s")
    else:
        raise KeyError("Aucun temps relatif trouvé (attendu: 't_ms' ou 't_s').")
    abs_time = (sim_start + dt).dt.tz_convert("UTC")
    return abs_time


def _close_tail_stop(
    df: pd.DataFrame,
    tail_window: int = 5,
    speed_col: str = "speed_mps",
    stop_col: str = "in_stop",
) -> pd.DataFrame:
    """Filet de sécurité : si la queue est un arrêt continu, on le clôt juste avant le dernier point,
    et on force un état terminal propre (plus d'arrêt/en livraison) sur la dernière ligne.
    """
    if stop_col not in df.columns or len(df) == 0:
        return df

    out = df.copy()
    tail = out.tail(tail_window)
    all_stopped = True
    if speed_col in out.columns:
        all_stopped = (tail[speed_col].abs() < 0.05).all()

    if all_stopped and tail[stop_col].all():
        if len(out) >= 2:
            last_idx = out.index[-1]
            prev_idx = out.index[-2]
            # Fin d'arrêt sur l'avant-dernier point
            out.at[prev_idx, stop_col] = False
            # Dernier point : état propre
            out.at[last_idx, stop_col] = False
            if "in_delivery" in out.columns:
                out.at[last_idx, "in_delivery"] = False
            if "delivery_state" in out.columns:
                out.at[last_idx, "delivery_state"] = "done"
            out.loc[last_idx, "run_state"] = "finished"
        else:
            # Série dégénérée
            last_idx = out.index[-1]
            out.at[last_idx, stop_col] = False
            if "in_delivery" in out.columns:
                out.at[last_idx, "in_delivery"] = False
            if "delivery_state" in out.columns:
                out.at[last_idx, "delivery_state"] = "done"
            out.loc[last_idx, "run_state"] = "finished"

    return out


@dataclass
class FlexisExporter:
    """Exporter autonome, utilisable comme Stage (interface minimale)."""

    name: str = "Exporter"

    def process(self, df: pd.DataFrame, ctx: Any) -> pd.DataFrame:
        # -------- Resolve config / output dir ----------
        cfg = getattr(ctx, "config", {}) or {}
        out_dir = (
            cfg.get("output", {}).get("dir")
            or cfg.get("export", {}).get("dir")
            or "data/simulations/default"
        )
        _ensure_dir(out_dir)

        export_cfg = cfg.get("export", {}) or {}
        fmt = (export_cfg.get("format") or "parquet").lower()
        filename = export_cfg.get("filename")
        if not filename:
            filename = "flexis.parquet" if fmt == "parquet" else "flexis.csv"
        events_filename = export_cfg.get("events_filename") or "events.jsonl"

        # --------- sim_start (tz-aware UTC) ------------
        sim_start = getattr(ctx, "sim_start", None)
        if sim_start is None:
            sim_start = _infer_sim_start_from_config(cfg)
            setattr(ctx, "sim_start", sim_start)

        # --------- timestamps absolus ------------------
        out = df.copy()
        abs_time = _compute_abs_time(out, sim_start)
        out["time_utc"] = abs_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # int64 ns → ms
        out["ts_ms"] = (abs_time.view("int64") // 1_000_000).astype("int64")

        # --------- fermeture arrêt de queue (option) ---
        if bool(export_cfg.get("close_tail_stop", True)):
            tail_window = int(export_cfg.get("tail_window", 5))
            out = _close_tail_stop(out, tail_window=tail_window)

        # --------- export table principale --------------
        fpath = os.path.join(out_dir, filename)
        if fmt == "csv":
            out.to_csv(fpath, index=False)
        elif fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
                out.to_parquet(fpath, index=False)
            except Exception:
                # fallback CSV si pyarrow absent
                fallback = os.path.splitext(fpath)[0] + ".csv"
                out.to_csv(fallback, index=False)
                fpath = fallback
        else:
            # format inconnu → csv
            fallback = os.path.splitext(fpath)[0] + ".csv"
            out.to_csv(fallback, index=False)
            fpath = fallback

        if hasattr(ctx, "logger"):
            ctx.logger.info(f"[Exporter] Wrote {fpath}")
        else:
            print(f"[Exporter] Wrote {fpath}")

        # --------- export éventuelle table d'événements -
        events = getattr(ctx, "events", None)
        if isinstance(events, (list, tuple)) and len(events) > 0:
            epath = os.path.join(out_dir, events_filename)
            with open(epath, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            if hasattr(ctx, "logger"):
                ctx.logger.info(f"[Exporter] Wrote {epath}")
            else:
                print(f"[Exporter] Wrote {epath}")

        return out


# Alias Stage si ton framework instancie par nom
Stage = FlexisExporter
