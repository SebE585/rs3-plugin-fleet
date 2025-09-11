# -*- coding: utf-8 -*-
import pandas as pd
from rs3_plugin_fleet.flexis.enricher import FlexisEnricher

# 1. Créer un DataFrame avec des timestamps "naive"
df_test = pd.DataFrame({
    "lat": [49.4432, 49.4435, 49.4438],
    "lon": [1.0999, 1.1002, 1.1005],
    "road_type": ["residential", "secondary", "primary"],
    "speed_mps": [10.0, 12.0, 8.0],
    "is_service": [False, True, False],
    "acc_z": [0.1, 2.6, 0.2],
    "timestamp": pd.date_range("2025-09-10 08:00:00", periods=3, freq="1s"),  # Timestamps naive
})

# 2. Configurer FlexisEnricher
config = {
    "traffic_profile": [
        {"from": "00:00", "to": "08:00", "level": "low"},
        {"from": "08:00", "to": "20:00", "level": "high"},
    ],
    "weather_timeline": [
        {"from": "00:00", "to": "12:00", "weather": "sunny"},
        {"from": "12:00", "to": "24:00", "weather": "rainy"},
    ],
    "infra_probability_per_km": {"bump": 0.1},
    "driver_events_rate_per_hour": {"harsh_brake": 0.5},
    "population_density_mode": "osm_heuristic",
    "timezone": "Europe/Paris",
}

# 3. Instancier FlexisEnricher
enricher = FlexisEnricher(config)

# 4. Créer un contexte minimal avec hz
class MockContext:
    def __init__(self, df):
        self.df = df
        self.hz = 1.0  # Fréquence en Hz

ctx = MockContext(df_test)

# 5. Appliquer l'enrichissement
result = enricher.run(ctx)

# 6. Afficher les résultats
if result.get("ok"):
    print("Colonnes après enrichissement :")
    print(ctx.df.columns.tolist())

    print("\nValeurs des colonnes flexis_* :")
    for col in ctx.df.columns:
        if col.startswith("flexis_"):
            print(f"{col}: {ctx.df[col].tolist()}")
else:
    print(f"Erreur lors de l'enrichissement : {result.get('msg')}")

# 7. Vérifier que les colonnes obligatoires sont présentes
required_cols = [
    "flexis_road_type",
    "flexis_road_curve_radius_m",
    "flexis_nav_event",
    "flexis_infra_event",
    "flexis_night",
    "flexis_driver_event",
    "flexis_traffic_level",
    "flexis_weather",
    "flexis_delivery_status",
    "flexis_population_density_km2",
]
missing = [c for c in required_cols if c not in ctx.df.columns]
if missing:
    print(f"\n❌ Colonnes manquantes : {missing}")
else:
    print("\n✅ Toutes les colonnes flexis_* sont présentes.")
