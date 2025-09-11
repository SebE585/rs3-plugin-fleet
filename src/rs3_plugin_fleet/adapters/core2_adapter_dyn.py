import logging
import importlib
from typing import Dict, Any, Tuple, List, Union
from copy import deepcopy
import inspect

logger = logging.getLogger(__name__)

# Silence very verbose OSMNX debug logs coming from RoadEnricher
OSMNX_LOGGER_NAME = "core2.stages.road_enricher"
def _silence_osmnx_debug() -> None:
    try:
        _lg = logging.getLogger(OSMNX_LOGGER_NAME)
        # Suppress DEBUG messages from the RoadEnricher module
        _lg.setLevel(logging.INFO)
        # Avoid double logging up the root chain at DEBUG level
        _lg.propagate = False
    except Exception:
        pass

class Stop:
    """Classe pour représenter un stop, compatible avec LegsPlan."""
    def __init__(self, type_: str, lat: float, lon: float, **kwargs):
        self.type = type_
        self.location = {"lat": lat, "lon": lon}
        self.is_depot = type_ == "depot"
        self.is_start = kwargs.get("is_start", False)
        self.is_end = kwargs.get("is_end", False)
        for k, v in kwargs.items():
            setattr(self, k, v)

def _stop_to_legsplan_dict(s: Any) -> Dict[str, Any]:
    """Convertit un stop en dictionnaire compatible avec LegsPlan."""
    if hasattr(s, "location"):
        lat = getattr(s, "location", {}).get("lat")
        lon = getattr(s, "location", {}).get("lon")
        t = getattr(s, "type", None)
        d = {
            "type": t or "delivery",
            "lat": float(lat) if lat is not None else 0.0,
            "lon": float(lon) if lon is not None else 0.0,
            "location": {"lat": lat, "lon": lon},
            "is_depot": getattr(s, "is_depot", t == "depot"),
            "is_start": getattr(s, "is_start", False),
            "is_end": getattr(s, "is_end", False),
        }
        for k in ("service_s", "name"):
            if hasattr(s, k):
                d[k] = getattr(s, k)
        return d
    if isinstance(s, dict):
        _t = s.get("type", "delivery")
        _lat = float(s.get("lat", 0.0))
        _lon = float(s.get("lon", 0.0))
        d = {
            "type": _t,
            "lat": _lat,
            "lon": _lon,
            "location": {"lat": _lat, "lon": _lon},
            "is_depot": s.get("is_depot", _t == "depot"),
            "is_start": s.get("is_start", False),
            "is_end": s.get("is_end", False),
        }
        for k in ("service_s", "name"):
            if k in s:
                d[k] = s[k]
        return d
    return {
        "type": "delivery",
        "lat": 0.0,
        "lon": 0.0,
        "location": {"lat": 0.0, "lon": 0.0},
        "is_depot": False,
        "is_start": False,
        "is_end": False,
    }

def _ensure_valid_stops(stops: List[Any]) -> List[Dict[str, Any]]:
    """S'assure que les stops sont valides pour LegsPlan."""
    if not stops:
        return []

    # Convertit tous les stops en dictionnaires compatibles
    plain_stops = [_stop_to_legsplan_dict(s) for s in stops]

    # S'assure qu'il y a au moins un stop de départ et un stop d'arrivée
    if len(plain_stops) == 1:
        first = plain_stops[0]
        end = {
            **first,
            "name": f"{first.get('name', 'DEPOT')}-END",
            "is_end": True,
            "is_depot": True,
            "service_s": 0,
        }
        plain_stops.append(end)

    # Force le premier stop à être un dépôt de départ
    if len(plain_stops) >= 1:
        plain_stops[0]["is_start"] = True
        plain_stops[0]["is_depot"] = True

    # Force le dernier stop à être un dépôt d'arrivée
    if len(plain_stops) >= 2:
        plain_stops[-1]["is_end"] = True
        plain_stops[-1]["is_depot"] = True

    return plain_stops

def _extract_stops_from_cfg(vcfg: Any) -> List[Any]:
    """Tente d'extraire une liste de stops à partir de différentes formes possibles de cfg.
    Retourne [] si rien n'est trouvé.
    """
    try:
        if not isinstance(vcfg, dict):
            return []
        # 1) stops au niveau racine
        if isinstance(vcfg.get("stops"), list):
            return vcfg.get("stops", [])
        # 2) cfg.first.route.stops
        first = vcfg.get("first") or {}
        route = first.get("route") or {}
        if isinstance(route.get("stops"), list):
            return route.get("stops", [])
        # 3) cfg.vehicles[0].stops ou cfg.vehicles[0].route.stops
        vehicles = vcfg.get("vehicles") or (vcfg.get("fleet") or {}).get("vehicles")
        if isinstance(vehicles, list) and vehicles:
            v0 = vehicles[0] or {}
            if isinstance(v0.get("stops"), list):
                return v0.get("stops", [])
            r = v0.get("route") or {}
            if isinstance(r.get("stops"), list):
                return r.get("stops", [])
        # 4) cfg.plan.route.stops
        plan = vcfg.get("plan") or {}
        r2 = plan.get("route") or {}
        if isinstance(r2.get("stops"), list):
            return r2.get("stops", [])
    except Exception:
        return []
    return []

class PreLegsBootstrap:
    """Prépare les stops pour LegsPlan."""
    def run(self, ctx):
        try:
            # Récupère les stops depuis le contexte
            stops = getattr(ctx, "stops", None)
            if not stops:
                # Essaye d'abord via ctx.vehicles ...
                vehicles = getattr(ctx, "vehicles", [])
                if vehicles and isinstance(vehicles, list):
                    try:
                        stops = vehicles[0].get("stops", [])
                    except Exception:
                        stops = None
            if not stops:
                # Puis inspecte la cfg pour trouver des stops
                vcfg = getattr(ctx, "cfg", None)
                extracted = _extract_stops_from_cfg(vcfg) if vcfg is not None else []
                if extracted:
                    stops = extracted

            # Normalise les stops pour LegsPlan
            if stops:
                valid_stops = _ensure_valid_stops(stops)
                setattr(ctx, "stops", valid_stops)
                # Alimente aussi ctx.cfg["stops"] dans le format attendu par LegsPlan
                try:
                    vcfg = getattr(ctx, "cfg", None)
                    if isinstance(vcfg, dict):
                        def _lp_stop(s: Dict[str, Any]) -> Dict[str, Any]:
                            lat = float(s.get("lat", s.get("location", {}).get("lat", 0.0)))
                            lon = float(s.get("lon", s.get("location", {}).get("lon", 0.0)))
                            out = {
                                "id": str(s.get("id", s.get("name", ""))),
                                "lat": lat,
                                "lon": lon,
                                "service_s": int(s.get("service_s", 0)),
                            }
                            # Fenêtres de temps optionnelles si présentes
                            if "tw_start" in s:
                                out["tw_start"] = s["tw_start"]
                            if "tw_end" in s:
                                out["tw_end"] = s["tw_end"]
                            return out
                        vcfg["stops"] = [_lp_stop(s) for s in valid_stops]
                    else:
                        # Si cfg n'est pas un dict, on essaie un attribut .stops si dispo
                        try:
                            setattr(vcfg, "stops", [
                                {
                                    "id": str(s.get("id", s.get("name", ""))),
                                    "lat": float(s.get("lat", s.get("location", {}).get("lat", 0.0))),
                                    "lon": float(s.get("lon", s.get("location", {}).get("lon", 0.0))),
                                    "service_s": int(s.get("service_s", 0)),
                                }
                                for s in valid_stops
                            ])
                        except Exception:
                            pass
                except Exception as _e:
                    logger.debug(f"[PreLegsBootstrap] Impossible d'injecter stops dans ctx.cfg: {_e}")
                # Tente aussi d'alimenter plan.stops et plan.route.stops SANS écraser les objets du contexte
                try:
                    plan = getattr(ctx, "plan", None)
                    if plan is not None:
                        # plan peut être un objet; on essaie d'utiliser setattr
                        try:
                            setattr(plan, "stops", list(valid_stops))
                        except Exception:
                            pass
                        # route peut vivre dans plan ou directement dans ctx
                        route = None
                        try:
                            route = getattr(plan, "route", None)
                        except Exception:
                            route = None
                        if route is None:
                            try:
                                route = getattr(ctx, "route", None)
                            except Exception:
                                route = None
                        if route is not None:
                            try:
                                setattr(route, "stops", list(valid_stops))
                            except Exception:
                                pass
                            try:
                                setattr(plan, "route", route)
                            except Exception:
                                pass
                except Exception as _e:
                    logger.debug(f"[PreLegsBootstrap] Impossible de propager stops dans plan/route: {_e}")
                logger.debug(f"[PreLegsBootstrap] Stops préparés pour LegsPlan: {valid_stops[:2]}...")

                # Assure: pas de dwell à la fin et start_time défini
                try:
                    if isinstance(valid_stops, list) and len(valid_stops) >= 2:
                        # Force aucun temps d'immobilisation au dernier stop
                        if isinstance(valid_stops[-1], dict):
                            valid_stops[-1]["service_s"] = 0
                            valid_stops[-1]["is_end"] = True
                            valid_stops[-1]["is_depot"] = True
                    # start_time_utc par défaut si manquant
                    vcfg = getattr(ctx, "cfg", None)
                    if isinstance(vcfg, dict) and not vcfg.get("start_time_utc"):
                        from datetime import datetime, timezone
                        vcfg["start_time_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                        # aussi refléter sur ctx si nécessaire
                        try:
                            setattr(ctx, "start_time_utc", vcfg["start_time_utc"])
                        except Exception:
                            pass
                    # Conseille au StopWaitInjector d'ignorer le dernier arrêt (si le stage le supporte)
                    if isinstance(vcfg, dict):
                        swi = vcfg.get("stopwait_injector", {}) or {}
                        # n'écrase pas l'existant, ajoute seulement l'option
                        swi.setdefault("skip_last", True)
                        vcfg["stopwait_injector"] = swi
                except Exception:
                    pass

            try:
                vcfg = getattr(ctx, "cfg", {})
                n = len(vcfg.get("stops", [])) if isinstance(vcfg, dict) else 0
                if n < 2:
                    logger.warning("[PreLegsBootstrap] Moins de 2 stops injectés dans ctx.cfg — LegsPlan échouera.")
                else:
                    logger.debug(f"[PreLegsBootstrap] {n} stop(s) injectés dans ctx.cfg pour LegsPlan")
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"[PreLegsBootstrap] Erreur: {e}")
            return False

def instantiate_stage(stage: Union[str, Dict[str, Any]]) -> Any:
    if isinstance(stage, str):
        if ":" not in stage:
            raise ValueError(f"Stage string doit être 'pkg.mod:Class' (reçu: {stage!r})")
        module_name, class_name = stage.split(":", 1)
        module = importlib.import_module(module_name)
        stage_class = getattr(module, class_name)
        return stage_class()
    if isinstance(stage, dict) and "class" in stage:
        module_name, class_name = stage["class"].split(":", 1)
        module = importlib.import_module(module_name)
        stage_class = getattr(module, class_name)
        params = {k: v for k, v in stage.items() if k != "class"}
        try:
            return stage_class(**params)
        except TypeError:
            return stage_class()
    raise ValueError(f"Stage mal formé: {stage!r}")

def build_pipeline_and_ctx(cfg: Dict[str, Any], sim_cfg: Dict[str, Any], config_path: str) -> Tuple[Any, Any]:
    # Ensure OSMNX/RoadEnricher DEBUG logs are silenced for this run
    _silence_osmnx_debug()

    """Construit le pipeline et le contexte."""
    stages = cfg.get("stages", [])
    # Essaie d'ajouter un stage d'altitude juste après RoadEnricher si disponible
    augmented_stages: List[Union[str, Dict[str, Any]]] = list(stages)
    altitude_symbols = [
        "core2.stages.altitude_enricher:AltitudeEnricher",
        "core2.stages.altitude:Altitude",
    ]
    altitude_symbol = None
    for sym in altitude_symbols:
        try:
            mod, cls = sym.split(":", 1)
            m = importlib.import_module(mod)
            getattr(m, cls)
            altitude_symbol = sym
            break
        except Exception:
            continue
    if altitude_symbol:
        try:
            idx = next(i for i, s in enumerate(augmented_stages)
                       if (isinstance(s, str) and s.endswith("road_enricher:RoadEnricher")) or
                          (isinstance(s, dict) and s.get("class", "").endswith("road_enricher:RoadEnricher")))
            augmented_stages.insert(idx + 1, {"class": altitude_symbol})
        except StopIteration:
            # S'il n'y a pas RoadEnricher, place-le avant IMUProjector sinon en fin
            try:
                idx2 = next(i for i, s in enumerate(augmented_stages)
                            if (isinstance(s, str) and s.endswith("imu_projector:IMUProjector")) or
                               (isinstance(s, dict) and s.get("class", "").endswith("imu_projector:IMUProjector")))
                augmented_stages.insert(idx2, {"class": altitude_symbol})
            except StopIteration:
                augmented_stages.append({"class": altitude_symbol})
    # On insert notre stage de bootstrap AVANT les stages déclarés, sous forme de symbole que le builder saura instancier.
    stages_cfg = [
        "rs3_plugin_fleet.adapters.core2_adapter_dyn:PreLegsBootstrap",
        *augmented_stages,
    ]

    # Pré-instancie tous les stages (fallback si le builder ne sait pas instancier)
    instantiated_stages: List[Any] = []
    for spec in stages_cfg:
        if isinstance(spec, str) or (isinstance(spec, dict) and "class" in spec):
            try:
                instantiated_stages.append(instantiate_stage(spec))
            except Exception as e:
                logger.error(f"[adapter] Échec instanciation du stage {spec!r}: {e}")
                raise
        else:
            instantiated_stages.append(spec)

    pipeline_cfg = deepcopy(cfg)
    pipeline_cfg["stages"] = stages_cfg

    pipeline_builder_symbol = cfg.get("contracts", {}).get("symbols", {}).get("pipeline_builder")
    if not pipeline_builder_symbol:
        raise ValueError("Pipeline builder symbol non trouvé dans la config.")

    module_name, func_name = pipeline_builder_symbol.split(':')
    module = importlib.import_module(module_name)
    build_func = getattr(module, func_name)
    # Toujours passer une *config* au builder pour qu'il instancie correctement les stages
    pipeline = build_func(pipeline_cfg)

    # Fallback: si le builder a renvoyé des stages bruts (dict), on retente avec des instances
    try:
        bad = False
        try:
            bad = any(isinstance(s, dict) for s in getattr(pipeline, "stages", []))
        except Exception:
            bad = False
        if bad:
            logger.debug("[adapter] Le builder a renvoyé des stages de type dict; tentative de reconstruction avec des instances...")
            # Essai 1: builder varargs
            rebuilt = None
            try:
                sig = inspect.signature(build_func)
                params = list(sig.parameters.values())
                if any(p.kind in (inspect.Parameter.VAR_POSITIONAL,) for p in params):
                    rebuilt = build_func(*instantiated_stages)
            except Exception:
                rebuilt = None
            # Essai 2: builder avec une liste
            if rebuilt is None:
                try:
                    rebuilt = build_func(instantiated_stages)
                except Exception:
                    rebuilt = None
            # Essai 3: construction directe du Pipeline core2
            if rebuilt is None:
                try:
                    from core2.pipeline import Pipeline
                    pipeline_name = cfg.get("name", "rs3-pipeline")
                    rebuilt = Pipeline(pipeline_name, instantiated_stages)
                except Exception as e:
                    logger.error(f"[adapter] Impossible de reconstruire un pipeline valide: {e}")
                    raise
            pipeline = rebuilt
    except Exception:
        # On laisse remonter: l'appelant affichera une erreur claire
        raise

    context_builder_symbol = cfg.get("contracts", {}).get("symbols", {}).get("context_builder", "core2.context:Context")
    module_name, func_name = context_builder_symbol.split(':')
    module = importlib.import_module(module_name)
    context_func = getattr(module, func_name)
    ctx = context_func(cfg)

    try:
        logger.debug(f"[adapter] Pipeline construit avec {len(pipeline.stages)} stage(s)")
    except Exception:
        logger.debug("[adapter] Pipeline construit (nombre de stages inconnu)")

    return pipeline, ctx
