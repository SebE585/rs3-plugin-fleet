# -*- coding: utf-8 -*-
"""
Generate a compact Flexis HTML report from timeline_flexis.csv.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple
import pandas as pd
from string import Template

def _segments_from_series(s: pd.Series) -> List[Tuple[str, int]]:
    if s is None or len(s) == 0:
        return []
    grp = (s != s.shift()).cumsum()
    out: List[Tuple[str, int]] = []
    for _, g in s.groupby(grp):
        v = str(g.iloc[0])
        out.append((v, len(g)))
    return out

def _cat_bar_html(title: str, segments: List[Tuple[str, int]], total_len: int, palette: Dict[str, str], label_map: Dict[str, str] | None = None) -> str:
    label_map = label_map or {}
    parts: List[str] = []
    for v, n in segments:
        key = str(v).lower()
        w = 100.0 * n / max(1, total_len)
        color = palette.get(key, "#ddd")
        label = label_map.get(key, str(v))
        parts.append(f'<div class="seg" style="width:{w:.6f}%;background:{color}" title="{label}"></div>')
    inner = "".join(parts)
    return f'''
      <div class="row">
        <div class="label">{title}</div>
        <div class="bar">{inner}</div>
      </div>
    '''

def _event_row_html(title: str, positions: List[Tuple[float, str]], icons: Dict[str, str]) -> str:
    parts: List[str] = []
    for left_pct, v in positions:
        key = str(v).lower()
        ico = icons.get(key, "•")
        parts.append(f'<div class="ev" style="left:{left_pct:.6f}%;" title="{v}">{ico}</div>')
    inner = "".join(parts)
    return f'''
      <div class="row">
        <div class="label">{title}</div>
        <div class="bar events">{inner}</div>
      </div>
    '''

def _positions_for(df: pd.DataFrame, col: str) -> List[Tuple[float, str]]:
    if col not in df.columns:
        return []
    s = df[col]
    N = len(s)
    out: List[Tuple[float, str]] = []
    for i, v in enumerate(s):
        if pd.isna(v):
            continue
        out.append((100.0 * i / max(1, N-1), str(v).lower()))
    return out

def write_flexis_report(outdir: str) -> str:
    """
    Build `flexis_report.html` next to timeline files in `outdir`.
    Returns the path to the HTML.
    """
    csv_path = os.path.join(outdir, "timeline_flexis.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"timeline_flexis.csv not found in {outdir}")
    df = pd.read_csv(csv_path, low_memory=False)

    N = len(df)
    if N == 0:
        raise ValueError("timeline_flexis.csv is empty")

    # --- Series prep --------------------------------------------------------
    def _num(name: str, default: float = float("nan")):
        return pd.to_numeric(df.get(name, default), errors="coerce")

    def _cat(name: str, default: str = "unknown"):
        s = df.get(name)
        if s is None:
            return pd.Series(default, index=df.index)
        return s.astype(str).str.lower().fillna(default)

    # Road curve buckets from radius
    r = _num("flexis_road_curve_radius_m")
    rb = pd.Series("unknown", index=df.index, dtype="object")
    rb = rb.mask(r < 30, "hairpin")
    rb = rb.mask((r >= 30) & (r < 150), "medium")
    rb = rb.mask(r >= 150, "straight")
    seg_curve = _segments_from_series(rb)

    seg_rt = _segments_from_series(_cat("flexis_road_type", "residential"))
    seg_w  = _segments_from_series(_cat("flexis_weather", "dry"))
    night = df.get("flexis_night")
    if night is None:
        night = pd.Series(False, index=df.index)
    seg_night = _segments_from_series(night.map({True:"night", False:"day"}))
    dens = _num("flexis_population_density_km2")
    db = pd.Series("unknown", index=df.index, dtype="object")
    db = db.mask(dens < 1500, "normal")
    db = db.mask((dens >= 1500) & (dens < 4000), "moderate")
    db = db.mask(dens >= 4000, "high")
    seg_dens = _segments_from_series(db)
    seg_t = _segments_from_series(_cat("flexis_traffic_level", "free"))

    # Events (positions)
    nav_pos = _positions_for(df, "flexis_nav_event")
    infra_pos = _positions_for(df, "flexis_infra_event")
    drv_pos = _positions_for(df, "flexis_driver_event")

    # Palettes / icons
    pal_curve = {"straight":"#b7e4c7","medium":"#e9f5a1","hairpin":"#f4a3a3","unknown":"#eee"}
    pal_rt = {
        "motorway":"#b7e4c7","primary":"#ffd580","secondary":"#ffe8a1","tertiary":"#f1f7c0",
        "residential":"#eef7d7","service":"#d7eff7","track":"#e0d4f7","parking_aisle":"#cde3f7","unknown":"#eee"
    }
    pal_w = {"dry":"#fffbe6","rain_light":"#dbeafe","rain_heavy":"#bfdbfe","unknown":"#eee"}
    pal_night = {"day":"#f2f2f2","night":"#000000"}
    pal_dens = {"normal":"#b7e4c7","moderate":"#f6f4a7","high":"#f4a3a3","unknown":"#eee"}
    pal_t = {"free":"#b7e4c7","moderate":"#f6f4a7","heavy":"#f4a3a3","unknown":"#eee"}
    nav_icons = {"left_turn":"↩", "right_turn":"↪", "u_turn":"⤵", "wait":"⏳", "stop":"⛔"}
    infra_icons = {"speed_hump":"⛰", "cobblestone":"🧱", "bridge_expansion":"⟷", "bump":"🟡", "roundabout":"⭕"}
    drv_icons = {"harsh_brake":"‼", "aggressive_accel":"➕", "sharp_lane_change":"⇄"}

    # --- Build rows ---------------------------------------------------------
    rows_html = ""
    rows_html += _cat_bar_html("Road curve", seg_curve, N, pal_curve, {"straight":"Straight","medium":"Medium","hairpin":"Hairpin"})
    rows_html += _cat_bar_html("Roadtype", seg_rt, N, pal_rt)
    rows_html += _event_row_html("Navigation events", nav_pos, nav_icons)
    rows_html += _event_row_html("Road infrastructure", infra_pos, infra_icons)
    rows_html += _cat_bar_html("Weather", seg_w, N, pal_w)
    rows_html += _cat_bar_html("Night conditions", seg_night, N, pal_night)
    rows_html += _cat_bar_html("Population density", seg_dens, N, pal_dens, {"normal":"Normal","moderate":"Moderate","high":"High"})
    rows_html += _cat_bar_html("Traffic", seg_t, N, pal_t)
    rows_html += _event_row_html("Driver events", drv_pos, drv_icons)

    # --- Template -----------------------------------------------------------
    tmpl_path = os.path.join(os.path.dirname(__file__), "templates", "flexis_report.html")
    with open(tmpl_path, "r", encoding="utf-8") as f:
        tmpl = Template(f.read())

    html = tmpl.safe_substitute(TITLE="Flexis — Timeline overview", ROWS=rows_html, SAMPLES=str(N))

    out_html = os.path.join(outdir, "flexis_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] Flexis HTML → {out_html}")
    return out_html