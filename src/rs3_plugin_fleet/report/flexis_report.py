# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
import os
import pandas as pd

DEFAULT_NAME = "flexis_report.html"

def _out_dir(cfg: Dict[str, Any]) -> str:
    return (cfg.get("output", {}) or {}).get("dir") or (cfg.get("export", {}) or {}).get("dir") or "data/simulations/default"

def _write_html(df: pd.DataFrame, out_dir: str, filename: str = DEFAULT_NAME) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    # mini-rapport autonome (table + quelques stats)
    total_pts = len(df)
    cols = list(df.columns)
    t0 = df["time_utc"].min() if "time_utc" in df else None
    t1 = df["time_utc"].max() if "time_utc" in df else None
    html_tbl = df.head(2000).to_html(index=False)  # évite fichiers monstrueux

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Flexis Report</title>
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:16px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:4px 6px}}</style>
</head>
<body>
<h1>Flexis Report</h1>
<p><b>Points:</b> {total_pts} — <b>Tps début:</b> {t0} — <b>Tps fin:</b> {t1}</p>
<p><b>Colonnes:</b> {', '.join(cols)}</p>
<h2>Aperçu données (max 2000 lignes)</h2>
{html_tbl}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

# API(s) de confort : le runner essaie ces fonctions
def generate_report(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if df is None or len(df) == 0: return
    p = _write_html(df, _out_dir(cfg))
    print(f"[Report] Flexis HTML → {p}")

def write_report(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    generate_report(df, cfg)

def build_report(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    generate_report(df, cfg)

def generate_html(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    generate_report(df, cfg)

class FlexisReport:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
    def run(self, df: pd.DataFrame, cfg: Dict[str, Any] | None = None):
        generate_report(df, cfg or self.cfg)
    def process(self, df: pd.DataFrame, cfg: Dict[str, Any] | None = None):
        self.run(df, cfg)
        return df