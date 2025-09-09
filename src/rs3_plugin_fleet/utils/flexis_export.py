# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers temps absolu/relatif
# ------------------------------------------------------------
def _compute_abs_time(df: pd.DataFrame, sim_start_utc) -> pd.Series:
    """
    Calcule l'heure absolue (UTC) à partir de:
      - t_ms (relatif) + sim_start, ou
      - t_s (relatif) + sim_start
    Clamp des deltas à ±100 ans pour éviter OutOfBoundsTimedelta.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    sim_start = pd.Timestamp(sim_start_utc).tz_convert("UTC")
    clip_win_ms = int(100 * 365.25 * 24 * 3600 * 1000)

    if "t_ms" in df.columns:
        t_ms = (
            pd.to_numeric(df["t_ms"], errors="coerce")
            .fillna(0)
            .clip(-clip_win_ms, clip_win_ms)
            .astype("int64")
        )
        dt = pd.to_timedelta(t_ms, unit="ms")
        return (pd.Series(sim_start, index=df.index) + dt).dt.tz_convert("UTC")

    if "t_s" in df.columns:
        t_s = pd.to_numeric(df["t_s"], errors="coerce").fillna(0)
        t_ms = (t_s * 1000.0).round().clip(-clip_win_ms, clip_win_ms).astype("int64")
        dt = pd.to_timedelta(t_ms, unit="ms")
        return (pd.Series(sim_start, index=df.index) + dt).dt.tz_convert("UTC")

    raise KeyError("Aucun temps relatif trouvé (attendu: 't_ms' ou 't_s').")


# ------------------------------------------------------------
# Trim fin de trace (arrêt prolongé)
# ------------------------------------------------------------
def _trim_tail_idle(df: pd.DataFrame, tail_window_s: float = 5.0) -> pd.DataFrame:
    """
    Supprime la traîne terminale “à l’arrêt” (accroche carte/arrêt prolongé).
    Heuristique:
      - on cherche le dernier index où la vitesse > 0.2 m/s OU un changement de position > ~2m
      - si la durée t_ms après cet index dépasse tail_window_s, on coupe après cet index
    """
    if df is None or len(df) == 0 or "t_ms" not in df.columns:
        return df

    v = None
    for c in ("speed_mps", "speed", "v"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            break
    if v is None:
        v = pd.Series(np.nan, index=df.index)

    # mouvement approximatif
    moved = pd.Series(False, index=df.index)
    if {"lat", "lon"}.issubset(df.columns):
        dlat = pd.to_numeric(df["lat"], errors="coerce").diff().abs()
        dlon = pd.to_numeric(df["lon"], errors="coerce").diff().abs()
        moved = ((dlat > 1e-5) | (dlon > 1e-5)).fillna(False)

    active = (v > 0.2) | moved
    if active.any():
        last_active_idx = active[active].index[-1]
    else:
        return df

    # Remplace fillna(method="ffill") par .ffill() (API moderne)
    t_ms = pd.to_numeric(df["t_ms"], errors="coerce").ffill().fillna(0)
    tail_len_ms = t_ms.iloc[-1] - t_ms.loc[last_active_idx]
    if tail_len_ms > (tail_window_s * 1000.0):
        return df.loc[:last_active_idx]
    return df


# ------------------------------------------------------------
# Résolution robuste de outdir / run_name depuis le ctx
# ------------------------------------------------------------
def _resolve_outdir(ctx, default_dir: str) -> str:
    """
    Essaie diverses conventions pour retrouver le même outdir que le core/rapports.
    Priorité:
      1) ctx.outdir / ctx.output_dir / ctx.out_dir
      2) ctx.output.dir (objet ou dict)
      3) ctx.config['output']['dir'] ou ctx.config['outdir']
      4) fallback: default_dir
    """
    # 1) attributs simples
    for attr in ("outdir", "output_dir", "out_dir"):
        val = getattr(ctx, attr, None)
        if isinstance(val, (str, Path)) and str(val).strip():
            return str(val)

    # 2) objet 'output' avec attribut/clé 'dir'
    output = getattr(ctx, "output", None)
    if output is not None:
        if isinstance(output, dict):
            val = output.get("dir")
            if isinstance(val, (str, Path)) and str(val).strip():
                return str(val)
        else:
            val = getattr(output, "dir", None)
            if isinstance(val, (str, Path)) and str(val).strip():
                return str(val)

    # 3) ctx.config dict
    config = getattr(ctx, "config", None)
    if isinstance(config, dict):
        out = config.get("output") or {}
        val = out.get("dir") or config.get("outdir")
        if isinstance(val, (str, Path)) and str(val).strip():
            return str(val)

    # 4) fallback
    return default_dir


def _resolve_run_name(ctx, default_filename: str) -> str:
    """
    Retrouve la même base de nom que le core:
      1) ctx.name / ctx.run_name
      2) ctx.config['name'] / ctx.config['run_name'] / ctx.config['output']['name']
      3) base du filename config (sans extension)
      4) 'run'
    """
    for attr in ("name", "run_name"):
        val = getattr(ctx, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    config = getattr(ctx, "config", None)
    if isinstance(config, dict):
        for key in ("name", "run_name"):
            val = config.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        out = config.get("output") or {}
        val = out.get("name")
        if isinstance(val, str) and val.strip():
            return val.strip()

    base = os.path.splitext(default_filename)[0] or "run"
    return base


# ------------------------------------------------------------
# Exporter
# ------------------------------------------------------------
@dataclass
class Config:
    out_dir: str
    filename: str = "flexis.csv"
    write_csv: bool = True
    write_parquet: bool = False
    write_events_jsonl: bool = False
    tail_window_s: float = 5.0


class Stage:
    """Exporter Flexis : écrit CSV/Parquet + colonnes temps absolu (time_utc, ts_ms).
    ⛳️ Chemin de sortie aligné sur le core/rapports :
        out_path = {ctx.outdir}/{ctx.name}-flexis.csv
    """
    name = "Exporter"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        cfg = cfg or {}
        export = (cfg.get("export") or {})
        out_dir = (cfg.get("output", {}) or {}).get("dir") or export.get("dir") or "data/simulations/default"
        filename = export.get("filename", "flexis.csv")
        tail_window = float(export.get("tail_window", 5))
        self.cfg = Config(out_dir=out_dir, filename=filename, tail_window_s=tail_window)

    def process(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        df = df.copy()

        # 1) temps absolu depuis t_ms/t_s + ctx.sim_start (robuste)
        sim_start = getattr(ctx, "sim_start", None) or pd.Timestamp.now(tz="UTC")
        abs_time = _compute_abs_time(df, sim_start)
        df["time_utc"] = abs_time
        df["ts_ms"] = (abs_time.astype("int64") // 1_000_000).astype("int64")

        # 2) trim de la queue “à l’arrêt”
        df = _trim_tail_idle(df, self.cfg.tail_window_s)

        # 3) Écriture alignée avec le core/rapports
        out_dir = _resolve_outdir(ctx, self.cfg.out_dir)
        run_name = _resolve_run_name(ctx, self.cfg.filename)

        # Évite le double suffixe si run_name termine déjà par '-flexis'
        out_basename = run_name if run_name.lower().endswith("-flexis") else f"{run_name}-flexis"
        out_path = Path(out_dir) / f"{out_basename}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cfg.write_parquet or str(out_path).lower().endswith(".parquet"):
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)

        print(f"[Exporter] Wrote {out_path}")

        # 4) remettre dans le ctx (chaînage éventuel)
        try:
            setattr(ctx, "df", df)
        except Exception:
            pass

        return df
