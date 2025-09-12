# -*- coding: utf-8 -*-
from __future__ import annotations

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


# ----------------------------
# Modèle Stop minimal
# ----------------------------
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


# ----------------------------
# Normalisation de stops
# ----------------------------
def _stop_to_legsplan_dict(s: Any) -> Dict[str, Any]:
    """Convertit un stop en dict compatible avec LegsPlan."""
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
        for k in ("service_s", "name", "id", "tw_start", "tw_end"):
            if hasattr(s, k):
                d[k] = getattr(s, k)
        return d

    if isinstance(s, dict):
        _t = s.get("type", "delivery")
        # Autoriser schémas {location:{lat,lon}} ou {lat,lon} plats
        _lat = s.get("lat", s.get("location", {}).get("lat", 0.0))
        _lon = s.get("lon", s.get("location", {}).get("lon", 0.0))
        d = {
            "type": _t,
            "lat": float(_lat),
            "lon": float(_lon),
            "location": {"lat": float(_lat), "lon": float(_lon)},
            "is_depot": s.get("is_depot", _t == "depot"),
            "is_start": s.get("is_start", False),
            "is_end": s.get("is_end", False),
        }
        for k in ("service_s", "name", "id", "tw_start", "tw_end"):
            if k in s:
                d[k] = s[k]
        return d

    # Fallback neutre
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
    """S'assure que la liste est exploitable par LegsPlan (>=2 stops, start/end marqués)."""
    if not stops:
        return []

    # Conversion uniforme
    plain = [_stop_to_legsplan_dict(s) for s in stops if s is not None]

    # Dupliquer si un seul stop fourni
    if len(plain) == 1:
        first = plain[0]
        end = {
            **first,
            "name": f"{first.get('name', 'DEPOT')}-END",
            "is_end": True,
            "is_depot": True,
            "service_s": 0,
        }
        plain.append(end)

    # Marquages bordures
    if len(plain) >= 1:
        plain[0]["is_start"] = True
        plain[0]["is_depot"] = True
    if len(plain) >= 2:
        plain[-1]["is_end"] = True
        plain[-1]["is_depot"] = True
        plain[-1]["service_s"] = 0  # pas d'immobilisation à l'arrivée

    # Id lisible par défaut
    for i, s in enumerate(plain):
        s.setdefault("id", s.get("name", f"STOP-{i:03d}"))

    return plain


# ----------------------------
# Extraction robuste depuis divers schémas YAML de flotte
# ----------------------------
def _extract_stops_from_cfg(vcfg: Any) -> List[Any]:
    """
    Essaie plusieurs chemins usuels. Retourne [] si rien trouvé.

    Schémas couverts (exemples) :
      - cfg.stops: [...]
      - cfg.first.route.stops
      - cfg.vehicles[0].stops
      - cfg.vehicles[0].route.stops
      - cfg.plan.route.stops
      - cfg.fleet.stops (liste globale)
    """
    try:
        if not isinstance(vcfg, dict):
            return []

        # 1) stops au niveau racine
        if isinstance(vcfg.get("stops"), list):
            return vcfg["stops"]

        # 2) cfg.first.route.stops
        first = vcfg.get("first") or {}
        route = first.get("route") or {}
        if isinstance(route.get("stops"), list):
            return route["stops"]

        # 3) cfg.vehicles[0].stops ou cfg.vehicles[0].route.stops
        vehicles = vcfg.get("vehicles") or (vcfg.get("fleet") or {}).get("vehicles")
        if isinstance(vehicles, list) and vehicles:
            v0 = vehicles[0] or {}
            if isinstance(v0.get("stops"), list):
                return v0["stops"]
            r = v0.get("route") or {}
            if isinstance(r.get("stops"), list):
                return r["stops"]

        # 4) cfg.plan.route.stops
        plan = vcfg.get("plan") or {}
        r2 = plan.get("route") or {}
        if isinstance(r2.get("stops"), list):
            return r2["stops"]

        # 5) cfg.fleet.stops (globale)
        fleet = vcfg.get("fleet") or {}
        if isinstance(fleet.get("stops"), list):
            return fleet["stops"]

    except Exception:
        return []

    return []


# ----------------------------
# Stage de bootstrap avant LegsPlan
# ----------------------------
class PreLegsBootstrap:
    """Prépare/normalise les stops puis les injecte dans ctx/cfg/plan/route pour LegsPlan."""
    def run(self, ctx):
        try:
            # 1) récup depuis ctx
            stops = getattr(ctx, "stops", None)

            # 2) tente ctx.vehicles[0].(stops|route.stops)
            if not stops:
                vehicles = getattr(ctx, "vehicles", [])
                if isinstance(vehicles, list) and vehicles:
                    v0 = vehicles[0] or {}
                    stops = v0.get("stops") or (v0.get("route") or {}).get("stops")

            # 3) extraction depuis cfg si besoin
            if not stops:
                vcfg = getattr(ctx, "cfg", None)
                extracted = _extract_stops_from_cfg(vcfg) if vcfg is not None else []
                if extracted:
                    stops = extracted

            # 4) normalisation / garde-fous
            if stops:
                valid_stops = _ensure_valid_stops(stops)

                # expose sur ctx
                setattr(ctx, "stops", valid_stops)

                # expose sur cfg (format compact pour LegsPlan si nécessaire)
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
                            if "tw_start" in s: out["tw_start"] = s["tw_start"]
                            if "tw_end" in s:   out["tw_end"] = s["tw_end"]
                            return out
                        vcfg["stops"] = [_lp_stop(s) for s in valid_stops]
                except Exception as _e:
                    logger.debug(f"[PreLegsBootstrap] Injection cfg.stops impossible: {_e}")

                # expose plan/route sans écraser
                try:
                    plan = getattr(ctx, "plan", None)
                    if plan is not None:
                        try: setattr(plan, "stops", list(valid_stops))
                        except Exception: pass

                        route = None
                        try: route = getattr(plan, "route", None)
                        except Exception: route = None
                        if route is None:
                            try: route = getattr(ctx, "route", None)
                            except Exception: route = None
                        if route is not None:
                            try: setattr(route, "stops", list(valid_stops))
                            except Exception: pass
                            try: setattr(plan, "route", route)
                            except Exception: pass
                except Exception as _e:
                    logger.debug(f"[PreLegsBootstrap] Propagation plan/route impossible: {_e}")

                # autres petits garde-fous utiles
                try:
                    from datetime import datetime, timezone
                    vcfg = getattr(ctx, "cfg", None)
                    if isinstance(vcfg, dict) and not vcfg.get("start_time_utc"):
                        vcfg["start_time_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                        try: setattr(ctx, "start_time_utc", vcfg["start_time_utc"])
                        except Exception: pass

                    # Conseil pour StopWaitInjector (si utilisé)
                    if isinstance(vcfg, dict):
                        swi = vcfg.get("stopwait_injector", {}) or {}
                        swi.setdefault("skip_last", True)
                        vcfg["stopwait_injector"] = swi
                except Exception:
                    pass

            # Sanity log
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


# ----------------------------
# Instanciation des stages & build
# ----------------------------

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

# ----------------------------
# Normalisation défensive du contenu pipeline.stages
# ----------------------------

def _normalize_pipeline_stages(pipeline) -> None:
    """Garantit que pipeline.stages est une liste plate d'instances avec .run.
    - Dé-neste le cas [[s1, s2, ...]] -> [s1, s2, ...]
    - Aplati toute sous-liste résiduelle
    - Instancie les specs (dict/str) via instantiate_stage
    """
    try:
        stages = getattr(pipeline, "stages", [])

        # Cas: [ [s1, s2, ...] ]
        if isinstance(stages, list) and len(stages) == 1 and isinstance(stages[0], list):
            stages = stages[0]

        # Aplatit récursivement les listes imbriquées
        flat: List[Any] = []
        def _add(x):
            if isinstance(x, list):
                for y in x:
                    _add(y)
            else:
                flat.append(x)
        _add(stages)

        # Instancie les specs dict/str → objets avec .run
        new_stages: List[Any] = []
        changed = False
        for s in flat:
            if hasattr(s, "run"):
                new_stages.append(s)
                continue
            if isinstance(s, (dict, str)):
                try:
                    new_stages.append(instantiate_stage(s))
                    changed = True
                    continue
                except Exception:
                    pass
            # autre type: garder tel quel
            new_stages.append(s)

        # Réécrit pipeline.stages si nécessaire
        if changed or flat is not getattr(pipeline, "stages", None):
            setattr(pipeline, "stages", new_stages)
    except Exception as _e:
        logger.debug(f"[adapter] Normalisation des stages ignorée: {_e}")


def build_pipeline_and_ctx(cfg: Dict[str, Any], sim_cfg: Dict[str, Any], config_path: str) -> Tuple[Any, Any]:
    # Réduire le bruit de RoadEnricher
    _silence_osmnx_debug()

    # 1) pipeline stages + insertion automatique d'un enrichisseur d'altitude si dispo
    stages = cfg.get("stages", [])
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
            try:
                idx2 = next(i for i, s in enumerate(augmented_stages)
                            if (isinstance(s, str) and s.endswith("imu_projector:IMUProjector")) or
                               (isinstance(s, dict) and s.get("class", "").endswith("imu_projector:IMUProjector")))
                augmented_stages.insert(idx2, {"class": altitude_symbol})
            except StopIteration:
                augmented_stages.append({"class": altitude_symbol})

    # Insert le bootstrap AVANT tout
    stages_cfg = [
        "rs3_plugin_fleet.adapters.core2_adapter_dyn:PreLegsBootstrap",
        *augmented_stages,
    ]

    # 2) pré-instanciation défensive (au cas où le builder renverrait des dicts)
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

    # 3) builder & context via symbols (par défaut core2_generic)
    pipeline_builder_symbol = cfg.get("contracts", {}).get("symbols", {}).get("pipeline_builder")
    if not pipeline_builder_symbol:
        raise ValueError("Pipeline builder symbol non trouvé dans la config.")

    module_name, func_name = pipeline_builder_symbol.split(':')
    module = importlib.import_module(module_name)
    build_func = getattr(module, func_name)

    # Toujours passer une *config* au builder
    pipeline = build_func(pipeline_cfg)
    _normalize_pipeline_stages(pipeline)

    # Si le builder a renvoyé des dicts pour stages, on tente une reconstruction
    try:
        bad = False
        try:
            bad = any(isinstance(s, dict) for s in getattr(pipeline, "stages", []))
        except Exception:
            bad = False

        if bad:
            logger.debug("[adapter] Le builder a renvoyé des stages de type dict; tentative de reconstruction avec des instances...")
            rebuilt = None

            # Essai 1: builder varargs
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

            # Essai 3: ne PAS importer core2 ici — on exige un builder valide
            if rebuilt is None:
                raise RuntimeError(
                    "Le builder fourni (contracts.symbols.pipeline_builder) n'a pas pu construire un pipeline valide. "
                    "Évitez tout import direct de core2 dans le plugin. Fournissez un builder dédié (ex: rs3_core2_adapter:build_pipeline) "
                    "qui s'occupera d'instancier le Pipeline côté AGPL."
                )

            pipeline = rebuilt
            _normalize_pipeline_stages(pipeline)
    except Exception:
        raise

    # Context (robuste, avec fallbacks)
    context_builder_symbol = cfg.get("contracts", {}).get("symbols", {}).get("context_builder") or "core2.context:Context"

    def _import_context_factory(symbol: str):
        mod_name, attr_name = symbol.split(':')
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Impossible d'importer '{mod_name}' (défini par contracts.symbols.context_builder). "
                f"Astuce: installez le paquet correspondant ou corrigez votre YAML."
            ) from e
        # Essai direct
        if hasattr(mod, attr_name):
            return getattr(mod, attr_name)
        # Fallbacks d'attributs courants
        for alt in ("Context", "make_context", "build_context", "build_ctx", "create_context"):
            if hasattr(mod, alt):
                logger.warning(
                    f"[adapter] Attribut '{attr_name}' introuvable dans {mod_name}; utilisation du fallback '{alt}'."
                )
                return getattr(mod, alt)
        # Pas de fallback core2 ici pour ne pas créer de dépendance dans le plugin
        raise AttributeError(
            f"Aucun fabriquant de contexte trouvé dans {mod_name}. "
            f"Exposez une fabrique (ex: build_context/Context) dans un paquet adapter côté core2 et référencez-la dans contracts.symbols.context_builder."
        )

    context_factory = _import_context_factory(context_builder_symbol)

    # Instanciation tolérante (fonction ou classe)
    try:
        ctx = context_factory(cfg)
    except TypeError:
        # Certains contextes n'acceptent pas d'arg; on injecte cfg après coup si possible
        ctx = context_factory()
        try:
            setattr(ctx, "cfg", cfg)
        except Exception:
            pass

    try:
        logger.debug(f"[adapter] Pipeline construit avec {len(pipeline.stages)} stage(s)")
    except Exception:
        logger.debug("[adapter] Pipeline construit (nombre de stages inconnu)")

    return pipeline, ctx