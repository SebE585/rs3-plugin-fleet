import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CustomLegsPlan:
    """Implémentation autonome de LegsPlan pour éviter les dépendances directes."""

    def run(self, ctx) -> Dict[str, Any]:
        """Exécute la logique de planification des legs."""
        try:
            stops = getattr(ctx, "stops", [])
            if not isinstance(stops, list) or len(stops) < 2:
                logger.error("CustomLegsPlan: Pas assez de stops valides")
                return {"ok": False, "msg": "Au moins 2 stops requis (départ et arrivée)."}

            # Vérifie que les stops ont les attributs requis
            for stop in stops:
                if not isinstance(stop, dict):
                    logger.error("CustomLegsPlan: Les stops doivent être des dictionnaires")
                    return {"ok": False, "msg": "Les stops doivent être des dictionnaires."}

                if "lat" not in stop or "lon" not in stop:
                    logger.error("CustomLegsPlan: Les stops doivent avoir 'lat' et 'lon'")
                    return {"ok": False, "msg": "Les stops doivent avoir 'lat' et 'lon'."}

            # Vérifie qu'il y a un stop de départ et un stop d'arrivée
            start_stops = [stop for stop in stops if stop.get("is_start", False)]
            end_stops = [stop for stop in stops if stop.get("is_end", False)]

            if len(start_stops) < 1 or len(end_stops) < 1:
                logger.error("CustomLegsPlan: Pas de stop de départ ou d'arrivée valide")
                return {"ok": False, "msg": "Au moins 2 stops requis (départ et arrivée)."}

            # Si tout est bon, construire un legs_plan minimal attendu par les stages suivants
            start_stop = start_stops[0] if start_stops else stops[0]
            end_stop = end_stops[-1] if end_stops else stops[-1]

            legs_plan = {
                "stops": stops,
                "start": start_stop,
                "end": end_stop,
                "count": len(stops),
            }

            # Construire une liste de legs minimale (chaînage stop i -> stop i+1)
            legs: List[Dict[str, Any]] = []
            for i in range(len(stops) - 1):
                legs.append({
                    "idx": i,
                    "from": stops[i],
                    "to": stops[i + 1],
                    # champs minimaux supplémentaires attendus par certains pipelines
                    "from_idx": i,
                    "to_idx": i + 1,
                    "mode": "drive",
                })

            # Publier le legs_plan et les legs dans le contexte pour LegsRoute et consorts
            try:
                setattr(ctx, "legs_plan", legs_plan)
            except Exception:
                pass
            try:
                setattr(ctx, "legs", legs)
            except Exception:
                pass
            try:
                plan_obj = getattr(ctx, "plan", None)
                if isinstance(plan_obj, dict):
                    plan_obj["legs_plan"] = legs_plan
                    plan_obj.setdefault("stops", stops)
                    plan_obj["legs"] = legs
                else:
                    setattr(ctx, "plan", {"legs_plan": legs_plan, "stops": stops, "legs": legs})
            except Exception:
                pass

            logger.info("CustomLegsPlan: Stops valides trouvés, planification réussie.")
            return {"ok": True, "msg": "Planification réussie.", "stops": stops, "legs_plan": legs_plan, "legs": legs}

        except Exception as e:
            logger.error(f"CustomLegsPlan: Erreur lors de la planification: {e}")
            return {"ok": False, "msg": f"Erreur lors de la planification: {e}"}
