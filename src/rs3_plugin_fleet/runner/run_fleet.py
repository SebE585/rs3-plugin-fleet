import argparse
import logging
import yaml
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_pipeline_and_ctx_dyn(config: Dict[str, Any], config_path: str) -> Tuple[Any, Any]:
    """Construit le pipeline et le contexte."""
    from rs3_plugin_fleet.adapters import core2_adapter_dyn as dyn
    return dyn.build_pipeline_and_ctx(cfg=config, sim_cfg=None, config_path=config_path)

def run_pipeline(pipeline, ctx):
    """Exécute le pipeline."""
    logger.info("Lancement du pipeline...")
    pipeline.run(ctx)

def main():
    parser = argparse.ArgumentParser(description="Lance une simulation de flotte.")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline, ctx = build_pipeline_and_ctx_dyn(config, args.config)
    run_pipeline(pipeline, ctx)

if __name__ == "__main__":
    main()
