# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
import importlib

from rs3_plugin_fleet.pipeline.adapter import build_pipeline_and_ctx
from rs3_plugin_fleet.utils.time_utils import ensure_time_contracts_ok

def _resolve_symbol(path: str):
    """
    Import 'pkg.mod:Attr' or 'pkg.mod.Attr' and return the symbol.
    """
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        parts = path.split(".")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)

def _call_loader_flex(load_fn, config_path: str):
    """
    Call loader in a tolerant way:
      - try with (config_path)
      - if TypeError indicates 0 positional args, retry with no arg
    """
    try:
        return load_fn(config_path)
    except TypeError as e:
        msg = str(e)
        if "0 positional arguments" in msg or "takes 0 positional arguments" in msg:
            return load_fn()
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Chemin du YAML de simulation")
    ap.add_argument("--load-config", required=True,
                    help="Symbole du chargeur de config (ex: core.config_loader:load_full_config)")
    # Optionnel: si dispo, on peut fournir un chargeur 'simulation only'
    ap.add_argument("--load-sim-config", default="core.config_loader:load_simulation_config",
                    help="Symbole du chargeur de config simulation (optionnel)")
    # Les options pipeline-class / builder sont désormais inutiles côté runner,
    # l'assemblage est géré dans build_pipeline_and_ctx, mais on les garde pour compat:
    ap.add_argument("--pipeline-class", default=None, help="(ignoré par le runner)")
    ap.add_argument("--pipeline-builder", default=None, help="(ignoré par le runner)")
    ap.add_argument("--with-altitude", action="store_true", default=False, help="(placeholder)")
    ap.add_argument("--with-flexis-export", action="store_true", default=False, help="(placeholder)")
    ns = ap.parse_args()

    # 1) Charger la config principale
    load_config = _resolve_symbol(ns.load_config)
    cfg = _call_loader_flex(load_config, ns.config)

    # 2) Charger la config 'simulation' si possible (optionnel)
    try:
        load_sim = _resolve_symbol(ns.load_sim_config) if ns.load_sim_config else None
    except Exception:
        load_sim = None
    sim_cfg = None
    if load_sim is not None:
        try:
            sim_cfg = _call_loader_flex(load_sim, ns.config)
        except Exception:
            sim_cfg = None

    # 3) Construire pipeline + contexte via l’adapter dynamique
    pipeline, ctx = build_pipeline_and_ctx(cfg, sim_cfg=sim_cfg, config_path=ns.config)

    # 4) Run + garde-fou temps
    result = pipeline.run(ctx)
    ensure_time_contracts_ok(result)
    print("[RUN] OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())