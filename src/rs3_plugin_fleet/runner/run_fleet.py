# -*- coding: utf-8 -*-
"""
Lance une simulation (mono ou multi-véhicules) en s'appuyant par défaut sur
le builder/context génériques core2_generic.py s'ils ne sont pas explicitement
définis dans la config (cfg.contracts.symbols.*).

Usage:
  python -m rs3_plugin_fleet.runner.run_fleet \
    --config path/to/fleet.yaml [--vehicle-id CCD-VL-01] [--list-vehicles]
"""

import argparse
import logging
import os
from copy import deepcopy
from typing import Dict, Any, Tuple, List

import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ----------------------------
# Chargement de la configuration
# ----------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Valeurs par défaut pour brancher core2_generic si rien n'est précisé
    contracts = cfg.setdefault("contracts", {})
    symbols = contracts.setdefault("symbols", {})
    symbols.setdefault("pipeline_builder", "rs3_contracts.core2_generic:build_pipeline")
    symbols.setdefault("context_builder", "rs3_contracts.core2_generic:build_context")

    return cfg


# ----------------------------
# Pont vers l'adapter dynamique
# ----------------------------
def build_pipeline_and_ctx_dyn(config: Dict[str, Any], config_path: str) -> Tuple[Any, Any]:
    """
    Construit le pipeline et le contexte via l'adapter dynamique.
    L'adapter honorera cfg.contracts.symbols.pipeline_builder/context_builder
    (par défaut: rs3_contracts.core2_generic).
    """
    from rs3_plugin_fleet.adapters import core2_adapter_dyn as dyn
    return dyn.build_pipeline_and_ctx(cfg=config, sim_cfg=None, config_path=config_path)


def run_pipeline(pipeline, ctx) -> None:
    """Exécute le pipeline."""
    logger.info("Lancement du pipeline...")
    pipeline.run(ctx)


# ----------------------------
# Post-traitement des stages d'export pour séparer par véhicule
# ----------------------------
def _update_stage_export_dirs_for_vehicle(stages: List[Any], vehicle_id: str) -> None:
    """
    Si le YAML contient des stages d'export, on réécrit leurs dossiers de sortie afin
    de séparer proprement par véhicule. MODIFIE `stages` in-place.

    - rs3_plugin_fleet.utils.flexis_export:Stage => config.export.dir += /<vehicle_id>
    - rs3_plugin_fleet.flexis.enricher:FlexisEnricher (si export_after_enrich) => export_outdir += /<vehicle_id>
    """
    if not isinstance(stages, list):
        return

    for s in stages:
        if not isinstance(s, dict):
            continue

        cls = s.get("class", "")

        # FlexisExporter utilitaire
        if cls.endswith("utils.flexis_export:Stage"):
            cfg = s.setdefault("config", {})
            exp = cfg.setdefault("export", {})
            base_dir = exp.get("dir", "data/simulations")
            exp["dir"] = os.path.join(base_dir, vehicle_id)

        # FlexisEnricher — export interne optionnel
        if cls.endswith("flexis.enricher:FlexisEnricher"):
            cfg = s.setdefault("config", {})
            if cfg.get("export_after_enrich"):
                outdir = cfg.get("export_outdir", "data/simulations/default")
                cfg["export_outdir"] = os.path.join(outdir, vehicle_id)


# ----------------------------
# Préparation d'une vue "mono-véhicule"
# ----------------------------
def _prepare_single_vehicle_cfg(cfg: Dict[str, Any], vehicle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construit une configuration dérivée ne contenant qu'un seul véhicule.
    Ajoute également un marquage `vehicle_id` à la racine pour que les stages
    puissent l'utiliser si besoin.
    """
    cfg_v = deepcopy(cfg)

    # Remplace la liste complète par uniquement ce véhicule
    cfg_v["vehicles"] = [vehicle]

    # Marqueur racine
    veh_id = str(vehicle.get("id", "vehicle"))
    cfg_v["vehicle_id"] = veh_id

    # Sépare les sorties par véhicule si possible
    stages = cfg_v.get("stages") or []
    _update_stage_export_dirs_for_vehicle(stages, veh_id)
    cfg_v["stages"] = stages

    return cfg_v


def _run_once(cfg: Dict[str, Any], config_path: str) -> None:
    """Construit et exécute un pipeline à partir d'une config déjà finalisée."""
    pipeline, ctx = build_pipeline_and_ctx_dyn(cfg, config_path)
    run_pipeline(pipeline, ctx)


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Lance une simulation de flotte.")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration YAML.")
    parser.add_argument("--vehicle-id", type=str, required=False, help="ID du véhicule à exécuter uniquement.")
    parser.add_argument("--list-vehicles", action="store_true", help="Lister les IDs de véhicules disponibles et quitter.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vehicles = cfg.get("vehicles") or []

    # Option: list vehicles and exit
    if args.list_vehicles:
        if not vehicles:
            print("No vehicles defined in configuration.")
        else:
            print("Available vehicles:")
            for v in vehicles:
                vid = str(v.get("id", "unnamed"))
                label = v.get("label") or v.get("name") or ""
                if label:
                    print(f" - {vid}: {label}")
                else:
                    print(f" - {vid}")
        return

    # Filtre éventuel sur un véhicule
    if args.vehicle_id is not None:
        filtered_vehicles = [v for v in vehicles if str(v.get("id")) == args.vehicle_id]
        if filtered_vehicles:
            vehicles = filtered_vehicles
            cfg["vehicles"] = vehicles
            cfg["vehicle_id"] = args.vehicle_id
        else:
            logger.warning(f"Vehicle ID '{args.vehicle_id}' not found in configuration vehicles.")
            vehicles = []

    # Cas simple: 0 ou 1 véhicule => exécution unique
    if len(vehicles) <= 1:
        _run_once(cfg, args.config)
        return

    # Multi-véhicules: boucle
    for v in vehicles:
        veh_id = str(v.get("id", "vehicle"))
        logger.info(f"[RUN] vehicle={veh_id}")
        cfg_v = _prepare_single_vehicle_cfg(cfg, v)
        _run_once(cfg_v, args.config)


if __name__ == "__main__":
    main()