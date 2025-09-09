# -*- coding: utf-8 -*-
"""
Adapter dynamique 'core2' simplifié — sans import statique.

- Résolution dynamique des classes core2 à l'exécution
- Pipeline par défaut robuste avec gestion d'erreurs
- Configuration flexible par YAML
- Code simplifié et plus maintenable

API exposée:
  - get_pipeline_cls() -> type
  - get_context_cls() -> type  
  - build_default_stages(cfg: dict) -> list[object]
"""

from __future__ import annotations
import os
import importlib
import traceback
from typing import Any, Dict, List, Optional

# Module "implémentation" par défaut (configurable via env)
CORE_MOD = os.environ.get("RS3_CORE_MODULE", "core2")

# Cache pour éviter les imports répétés
_CACHE: Dict[str, Any] = {}


def _resolve_symbol(path: str) -> Any:
    """Résout dynamiquement un symbole module.classe"""
    if path in _CACHE:
        return _CACHE[path]
    
    try:
        mod_path, _, name = path.rpartition(".")
        if not mod_path or not name:
            raise ImportError(f"Invalid symbol path: {path}")
        
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, name)
        _CACHE[path] = obj
        print(f"[ADAPTER] Resolved {path} ✓")
        return obj
        
    except Exception as e:
        print(f"[ADAPTER] Failed to resolve {path}: {e}")
        raise


def _safe_resolve(paths: List[str]) -> Optional[Any]:
    """Essaie plusieurs chemins, retourne le premier qui fonctionne"""
    for path in paths:
        try:
            return _resolve_symbol(path)
        except Exception:
            continue
    return None


def _safe_instantiate(cls: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """Instancie une classe avec gestion d'erreurs robuste"""
    if cls is None:
        return None
    
    config = config or {}
    stage_name = getattr(cls, '__name__', str(cls))
    
    try:
        # Essaie avec la config
        instance = cls(**config)
        print(f"[ADAPTER] Created {stage_name} with config ✓")
        return instance
    except Exception:
        try:
            # Essaie sans config
            instance = cls()
            print(f"[ADAPTER] Created {stage_name} (no config) ✓")
            return instance
        except Exception as e:
            print(f"[ADAPTER] Failed to create {stage_name}: {e}")
            return None


# ---------- API principale --------------------------------------------------

def get_pipeline_cls():
    """Retourne la classe PipelineSimulator"""
    return _resolve_symbol(f"{CORE_MOD}.pipeline.PipelineSimulator")


def get_context_cls():
    """Retourne la classe Context"""  
    return _resolve_symbol(f"{CORE_MOD}.context.Context")


def build_default_stages(cfg: Dict[str, Any]) -> List[Any]:
    """Construit le pipeline par défaut avec stages essentiels"""
    print(f"[ADAPTER] Building default pipeline from {CORE_MOD}")
    
    stages = []
    
    # Stage definitions: (name, module, default_config)
    stage_definitions = [
        # 1. Planning & Routing (obligatoires)
        ("LegsPlan", "legs_plan.LegsPlan", {}),
        ("LegsRoute", "legs_route.LegsRoute", {
            "profile": cfg.get("osrm", {}).get("profile", "driving")
        }),
        ("LegsStitch", "legs_stitch.LegsStitch", {}),
        
        # 2. Enrichissement 
        ("RoadEnricher", "road_enricher.RoadEnricher", {}),
        ("GeoSpikeFilter", "geo_spike_filter.GeoSpikeFilter", 
         cfg.get("geo_spike_filter", {"vmax_kmh": 160.0, "hard_jump_m": 500.0})),
        ("LegsRetimer", "legs_retimer.LegsRetimer", 
         cfg.get("legs_retimer", {"default_kmh": 50, "min_dt": 0.05})),
        
        # 3. Stops & Smoothing
        ("StopWaitInjector", "stopwait_injector.StopWaitInjector", 
         cfg.get("stop_wait_injector", {"tail_wait_s": 0})),
        ("StopSmoother", "stop_smoother.StopSmoother", 
         cfg.get("stop_smoother", {
             "v_in": 0.25, "t_in": 2.0, "v_out": 0.6, "t_out": 2.5, "lock_pos": False
         })),
        
        # 4. IMU & Sensors
        ("IMUProjector", "imu_projector.IMUProjector", cfg.get("imu_projector", {})),
        ("NoiseInjector", "noise_injector.NoiseInjector", cfg.get("noise", {})),
        
        # 5. Sync & Validation (ordre important!)
        ("SpeedSync", "speed_sync.SpeedSync", 
         cfg.get("speed_sync", {"keep_start_zero": True, "head_window_s": 2.0})),
        ("Validators", "validators.Validators", {}),
        ("Exporter", "exporter.Exporter", {}),
    ]
    
    # Optional lockers (try grouped first, then individual)
    locker_definitions = [
        ("InitialStopLocker", ["stop_lockers.InitialStopLocker", "initial_stop_locker.InitialStopLocker"]),
        ("MidStopsLocker", ["stop_lockers.MidStopsLocker", "mid_stops_locker.MidStopsLocker"]),  
        ("FinalStopLocker", ["stop_lockers.FinalStopLocker", "final_stop_locker.FinalStopLocker"]),
    ]
    
    # Build main stages
    for stage_name, module_path, stage_config in stage_definitions:
        full_path = f"{CORE_MOD}.stages.{module_path}"
        
        try:
            cls = _resolve_symbol(full_path)
            instance = _safe_instantiate(cls, stage_config)
            if instance is not None:
                stages.append(instance)
            else:
                print(f"[ADAPTER] ⚠️  Skipped {stage_name} (instantiation failed)")
                
        except Exception as e:
            if stage_name in ["LegsPlan", "LegsRoute", "LegsStitch", "Exporter"]:
                print(f"[ADAPTER] ❌ CRITICAL: {stage_name} failed: {e}")
                raise  # Ces stages sont critiques
            else:
                print(f"[ADAPTER] ⚠️  Optional stage {stage_name} failed: {e}")
    
    # Add optional lockers (insert after StopSmoother)
    locker_stages = []
    for locker_name, paths in locker_definitions:
        full_paths = [f"{CORE_MOD}.stages.{p}" for p in paths]
        cls = _safe_resolve(full_paths)
        if cls:
            instance = _safe_instantiate(cls)
            if instance:
                locker_stages.append(instance)
    
    # Insert lockers after StopSmoother
    if locker_stages:
        insert_pos = len(stages) - 3  # Before SpeedSync, Validators, Exporter
        for i, stage in enumerate(stages):
            if hasattr(stage, '__class__') and 'StopSmoother' in stage.__class__.__name__:
                insert_pos = i + 1
                break
        
        stages[insert_pos:insert_pos] = locker_stages
        locker_names = [s.__class__.__name__ for s in locker_stages]
        print(f"[ADAPTER] Inserted lockers: {' → '.join(locker_names)}")
    
    # Final pipeline summary
    stage_names = []
    for stage in stages:
        if stage is not None:
            name = getattr(stage, 'name', stage.__class__.__name__)
            stage_names.append(name)
    
    print(f"[ADAPTER] Final pipeline: {' → '.join(stage_names)}")
    print(f"[ADAPTER] Total stages: {len(stages)}")
    
    # Validation critique
    critical_stages = ["LegsPlan", "LegsRoute", "LegsStitch", "Exporter"]
    missing_critical = []
    for critical in critical_stages:
        if not any(critical in name for name in stage_names):
            missing_critical.append(critical)
    
    if missing_critical:
        raise RuntimeError(f"CRITICAL stages missing: {missing_critical}")
    
    return stages


def build_pipeline_and_context(cfg: Dict[str, Any]) -> tuple[Any, Any]:
    """Fonction complète pour construire pipeline + contexte (utilisée par runner)"""
    print(f"[ADAPTER] build_pipeline_and_context called")
    
    try:
        # 1. Get classes
        PipelineCls = get_pipeline_cls()
        ContextCls = get_context_cls()
        
        # 2. Build stages
        stages = build_default_stages(cfg)
        
        # 3. Create pipeline
        pipeline = PipelineCls(stages)
        
        # 4. Create context
        try:
            ctx = ContextCls(cfg)
        except Exception:
            ctx = ContextCls()
            
        # 5. Inject config into context
        try:
            ctx.config = cfg
        except Exception:
            setattr(ctx, 'config', cfg)
            
        print(f"[ADAPTER] ✅ Successfully built pipeline with {len(stages)} stages")
        return pipeline, ctx
        
    except Exception as e:
        print(f"[ADAPTER] ❌ build_pipeline_and_context failed: {e}")
        traceback.print_exc()
        raise