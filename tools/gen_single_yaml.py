# tools/gen_single_yaml.py
# Génère un seul YAML complet : profils, stages, et 20 véhicules (15 colis, 5 meubles)
import os, sys, random, math, yaml
from datetime import datetime

random.seed(42)

# --- paramètres globaux ---
OUT_PATH = "src/rs3_plugin_fleet/config/coin-coin-delivery.generated.yaml"

DEPOT_LAT, DEPOT_LON = 49.3210, 1.2910
DEPOT = {"name": "DEPOT-START-ROMILLY", "lat": DEPOT_LAT, "lon": DEPOT_LON, "service_s": 0}
ROLLIN = {"name": "ROLL-IN-ROMILLY",     "lat": DEPOT_LAT, "lon": DEPOT_LON, "service_s": 60}

# centres urbains Haute-Normandie (coord réalistes)
CITIES = [
    ("rouen",        49.4431, 1.0993),
    ("lehavre",      49.4944, 0.1079),
    ("dieppe",       49.9229, 1.0770),
    ("evreux",       49.0280, 1.1500),
    ("vernon",       49.0930, 1.4890),
    ("fecamp",       49.7570, 0.3750),
    ("letreport",    50.0590, 1.3760),
    ("yvetot",       49.6170, 0.7530),
    ("louviers",     49.2160, 1.1710),
    ("gisors",       49.2810, 1.7770),
    ("elbeuf",       49.2770, 1.0130),
    ("pontaudemer",  49.3530, 0.5120),
    ("lesandelys",   49.2460, 1.4130),
    ("bolbec",       49.5730, 0.4740),
    ("barentin",     49.5460, 0.9530),
]  # 15 tournées colis

# 5 zones meubles (plus éparses)
FURN_CENTERS = [
    ("furniture-lyons",  49.4010, 1.4690),
    ("furniture-etretat",49.7060, 0.2040),
    ("furniture-bernay", 49.0930, 0.5950),
    ("furniture-eu",     50.0470, 1.4180),
    ("furniture-gisors", 49.1640, 1.3380),
]

def rand_point_near(lat0: float, lon0: float, radius_km: float) -> tuple[float,float]:
    """
    Point aléatoire uniforme dans un disque de rayon `radius_km` autour de (lat0, lon0).
    1°lat ~ 111 km ; 1°lon ~ 111 km * cos(lat).
    """
    r = radius_km * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()
    dlat = r / 111.0
    dlon = r / (111.0 * math.cos(math.radians(lat0)))
    return (lat0 + dlat * math.sin(theta), lon0 + dlon * math.cos(theta))

def round_pt(pt):
    lat, lon = pt
    return round(lat, 6), round(lon, 6)

def make_parcel_vehicle(idx: int, slug: str, lat0: float, lon0: float) -> dict:
    """
    50–70 livraisons réalistes autour d’une ville (+ communes proches).
    Rayon 3.5–5.0 km pour densité urbaine/suburbaine crédible.
    """
    dwell_s = 120
    nstops = random.randint(50, 70)
    radius = random.uniform(3.5, 5.0)

    stops = [DEPOT.copy()]
    for k in range(1, nstops + 1):
        lat, lon = rand_point_near(lat0, lon0, radius_km=radius)
        lat, lon = round_pt((lat, lon))
        stops.append({"name": f"{slug.upper()}-{k:03d}", "lat": lat, "lon": lon, "service_s": dwell_s})
    stops.append(ROLLIN.copy())

    return {"id": f"CCD-VL-{idx:02d}", "profile": "parcels_70x2min", "stops": stops}

def make_furniture_vehicle(idx: int, slug: str, lat0: float, lon0: float) -> dict:
    """
    5 livraisons espacées (~10–20 km) pour simuler des tournées longues (1h par stop).
    """
    dwell_s = 3600
    nstops = 5
    stops = [DEPOT.copy()]

    # génère 5 points plus dispersés (rayon 10–20 km)
    for k in range(1, nstops + 1):
        lat, lon = rand_point_near(lat0, lon0, radius_km=random.uniform(10.0, 20.0))
        lat, lon = round_pt((lat, lon))
        stops.append({"name": f"{slug.upper()}-{k:03d}", "lat": lat, "lon": lon, "service_s": dwell_s})

    stops.append(ROLLIN.copy())
    return {"id": f"CCD-FUR-{idx:02d}", "profile": "furniture_sparse_1h", "stops": stops}

def build_full_config() -> dict:
    # En-tête + symboles (identiques à la conf existante)
    cfg = {
        "client": "Coin-Coin Delivery",
        "contracts": {
            "symbols": {
                "load_config":      "core.config_loader:load_config",
                "pipeline_builder": "core2.pipeline:build_pipeline",
                "pipeline_class":   "core2.pipeline:PipelineSimulator",
            }
        },
        "osrm": {"profile": "driving", "base_url": "http://localhost:5003"},
        "profiles": {
            "parcels_70x2min": {
                "hz": 10,
                "osrm": {"profile": "driving", "base_url": "http://localhost:5003"},
                "stops_per_vehicle": 70,
                "dwell_seconds": {"mean": 120, "jitter": 0.2},
                "speed_targets": {"residential": 30, "primary": 50, "motorway": 110},
                "traffic_profile": [
                    {"from": "08:00", "to": "09:30", "level": "heavy"},
                    {"from": "09:30", "to": "16:30", "level": "moderate"},
                    {"from": "16:30", "to": "18:30", "level": "heavy"},
                    {"from": "18:30", "to": "23:00", "level": "free"},
                ],
                "weather_timeline": [
                    {"from": "08:00", "to": "11:00", "weather": "dry"},
                    {"from": "11:00", "to": "13:00", "weather": "rain_light"},
                    {"from": "13:00", "to": "20:00", "weather": "dry"},
                ],
            },
            "furniture_sparse_1h": {
                "hz": 10,
                "osrm": {"profile": "driving", "base_url": "http://localhost:5003"},
                "stops_per_vehicle": 10,
                "dwell_seconds": {"mean": 3600, "jitter": 0.1},
            },
        },
    }

    # véhicules
    vehicles = []
    # 15 colis
    for i, (slug, lat, lon) in enumerate(CITIES, start=1):
        vehicles.append(make_parcel_vehicle(i, slug, lat, lon))
    # 5 meubles
    for j, (slug, lat, lon) in enumerate(FURN_CENTERS, start=1):
        vehicles.append(make_furniture_vehicle(j, slug, lat, lon))

    cfg["vehicles"] = vehicles

    # reste de la config (copié/aligné sur ta conf actuelle)
    cfg.update({
        "road_enrich": {
            "stream_url_base": "http://localhost:5002/nearest_road_batch_stream",
            "batch_size": 400,
            "search_radius_m": 120,
            "fallback_radius_m": 220,
            "max_retries": 3,
            "connect_timeout_s": 6.0,
            "read_timeout_s": 25.0,
            "request_timeout_s": 30.0,
        },
        "speed_sync": {
            "keep_start_zero": True,
            "head_window_s": 2.0,
            "lat_acc_max_mps2": 2.2,
            "corner_speed_cap": True,
            "v_min_mps": 0.3,
        },
        "geo_spike_filter": {
            "vmax_kmh": 160.0,
            "hard_jump_m": 500.0,
            "max_delta_heading_deg": 28,
            "min_speed_mps": 0.5,
        },
        "stop_smoother": {"window_s": 2.0},
        "legs_retimer": {
            "speed_by_type": {
                "motorway": 110, "trunk": 100, "primary": 70, "secondary": 50,
                "tertiary": 40, "residential": 30, "service": 20,
                "unclassified": 50, "unknown": 50
            },
            "default_kmh": 50,
            "use_column_target_speed": True,
            "min_dt": 0.05,
            "hz": 10.0,
        },
        "flexis": {
            "infra_probability_per_km": {"residential": 0.05, "primary": 0.01},
            "driver_events_rate_per_hour": {"harsh_brake": 2.0, "aggressive_accel": 2.0},
        },
        "stages": [
            {"class": "core2.stages.legs_plan:LegsPlan"},
            {"class": "core2.stages.legs_route:LegsRoute"},
            {"class": "core2.stages.legs_stitch:LegsStitch"},
            {"class": "core2.stages.road_enricher:RoadEnricher"},
            {"class": "rs3_plugin_altitude_agpl.plugin:AltitudeStage"},
            {"class": "core2.stages.geo_spike_filter:GeoSpikeFilter"},
            {"class": "core2.stages.legs_retimer:LegsRetimer"},
            {"class": "core2.stages.stopwait_injector:StopWaitInjector"},
            {"class": "core2.stages.stop_smoother:StopSmoother"},
            {"class": "core2.stages.initial_stop_locker:InitialStopLocker"},
            {"class": "core2.stages.mid_stops_locker:MidStopsLocker"},
            {"class": "core2.stages.final_stop_locker:FinalStopLocker"},
            {"class": "core2.stages.imu_projector:IMUProjector"},
            {"class": "core2.stages.noise_injector:NoiseInjector"},
            {"class": "core2.stages.speed_sync:SpeedSync"},
            {"class": "core2.stages.validators:Validators"},
            {
                "class": "rs3_plugin_fleet.flexis.enricher:FlexisEnricher",
                "config": {
                    "traffic_profile": [
                        {"from": "08:00", "to": "09:30", "level": "heavy"},
                        {"from": "09:30", "to": "16:30", "level": "moderate"},
                        {"from": "16:30", "to": "18:30", "level": "heavy"},
                        {"from": "18:30", "to": "23:00", "level": "free"},
                    ],
                    "weather_timeline": [
                        {"from": "08:00", "to": "11:00", "weather": "dry"},
                        {"from": "11:00", "to": "13:00", "weather": "rain_light"},
                        {"from": "13:00", "to": "20:00", "weather": "dry"},
                    ],
                    "force_departure_after_wait": True,
                    "infra_probability_per_km": {"residential": 0.05, "primary": 0.01},
                    "driver_events_rate_per_hour": {"harsh_brake": 2.0, "aggressive_accel": 2.0},
                    "population_density_mode": "osm_heuristic",
                    "export_after_enrich": True,
                    "export_outdir": "data/simulations/default",
                    "export_filename": "flexis_final.csv",
                },
            },
            {
                "class": "rs3_plugin_fleet.utils.flexis_export:Stage",
                "config": {
                    "export": {
                        "dir": "data/simulations/ccd_parcels_rouen",
                        "filename": "flexis_final",
                        "tail_window": 5,
                    },
                    "formats": {"csv": True, "parquet": False},
                    "report": {"enabled": True, "filename": "flexis_report.html"},
                },
            },
            {"class": "core2.stages.exporter:Exporter"},
        ],
        "exporter": {
            "report": {
                "enabled": True,
                "standalone_map": True,
                "inline_data": True,
                "map_sample_hz": 20.0,
                "charts_sample_hz": 2.0,
                "break_on_nan": True,
                "max_segment_gap_s": 1.0,
            }
        },
        "validators": {
            "cadence": {"expected_hz": 10.0, "tolerance_ppm": 10000, "enabled": True},
            "dynamics": {"turn_radius_min_m": 10, "lateral_consistency_tol": 0.2},
        },
    })

    return cfg

def main():
    cfg = build_full_config()

    # crée le dossier si besoin et écrit le fichier
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # also dump to stdout to pouvoir rediriger si souhaité
    sys.stdout.write(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    sys.stderr.write(f"[OK] YAML complet écrit dans {OUT_PATH} ({datetime.now().isoformat(timespec='seconds')})\n")

if __name__ == "__main__":
    main()