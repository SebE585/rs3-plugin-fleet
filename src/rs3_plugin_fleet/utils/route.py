# src/rs3_plugin_fleet/utils/route.py
from __future__ import annotations
import math
from typing import Any, Dict, List

_DEF_EPS = 1e-6

def collect_pois(zones: List[Dict[str, Any]], allowed_zones: List[str] | None = None) -> List[Dict[str, Any]]:
    allowed = set(allowed_zones) if allowed_zones else None
    pois: List[Dict[str, Any]] = []
    if not isinstance(zones, list):
        return pois
    for z in zones:
        zname = z.get("name")
        if allowed and zname not in allowed:
            continue
        for p in (z.get("poi") or []):
            pois.append({
                "name": p.get("name") or f"POI-{len(pois)+1}",
                "lat": float(p["lat"]),
                "lon": float(p["lon"]),
                "zone": zname,
            })
    return pois

def plan_stops_from_zones(zones, n, seed_offset=0, allowed_zones=None):
    n = int(max(0, n))
    pois = collect_pois(zones, allowed_zones=allowed_zones)
    if len(pois) == 0:
        return []
    if len(pois) == 1:
        p = pois[0]
        return [
            {"lat": p["lat"], "lon": p["lon"], "name": p["name"]},
            {"lat": p["lat"] + 1e-4, "lon": p["lon"] + 1e-4, "name": p["name"] + "-B"},
        ][: max(2, n)]
    out: List[Dict[str, Any]] = []
    idx = seed_offset % len(pois)
    while len(out) < max(2, n):
        p = pois[idx]
        out.append({"lat": p["lat"], "lon": p["lon"], "name": p["name"]})
        idx = (idx + 1) % len(pois)
    return out[: max(2, n)]

def _same_coord(a: dict, b: dict, eps: float = _DEF_EPS) -> bool:
    try:
        return abs(float(a["lat"]) - float(b["lat"])) < eps and abs(float(a["lon"]) - float(b["lon"])) < eps
    except Exception:
        return False

def _meters_to_deg(lat: float, meters: float) -> tuple[float, float]:
    dlat = meters / 111_320.0
    dlon = meters / (111_320.0 * max(1e-6, math.cos(math.radians(lat))))
    return dlat, dlon

def ensure_roundtrip_with_depot(stops, depot, roll_eps_m: float = 0.0, final_service_s: float | None = None):
    if not depot:
        return stops
    dep = {
        "name": depot.get("name", "DEPOT"),
        "lat": float(depot["lat"]),
        "lon": float(depot["lon"]),
        "service_s": float(depot.get("service_s", 0.0)),
    }
    out: list[dict] = []
    out = [dict(dep), dict(dep)] if not stops else [s.copy() for s in stops]

    if not _same_coord(out[0], dep):
        start_dep = dict(dep); start_dep["name"] = dep.get("name") or "DEPOT-START"
        start_dep.setdefault("service_s", 0.0)
        out.insert(0, start_dep)

    if roll_eps_m and roll_eps_m > 0:
        dlat, dlon = _meters_to_deg(dep["lat"], roll_eps_m)
        out.insert(1, {"name": "ROLL-OUT","lat": dep["lat"] + dlat,"lon": dep["lon"] + dlon,"service_s": 0.0})

    if not _same_coord(out[-1], dep):
        end_dep = dict(dep); end_dep["name"] = (dep.get("name") or "DEPOT") + "-END"
        if final_service_s is not None: end_dep["service_s"] = float(final_service_s)
        out.append(end_dep)
    else:
        out[-1]["service_s"] = float(final_service_s) if final_service_s is not None else float(out[-1].get("service_s", 0.0))

    if roll_eps_m and roll_eps_m > 0:
        dlat, dlon = _meters_to_deg(dep["lat"], roll_eps_m)
        out.insert(-1, {"name": "ROLL-IN","lat": dep["lat"] - dlat,"lon": dep["lon"] - dlon,"service_s": 0.0})
    return out