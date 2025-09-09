# Legal Notes — RS3 Plugin Fleet

## 📜 Objectif
Ce document explique le positionnement juridique du plugin **Fleet** vis-à-vis des dépendances
du projet **RoadSimulator3 (RS3)** et des licences impliquées.

## 1. Licences concernées

- **RS3-core2** : publié sous **AGPL-3.0-only**  
- **rs3-contracts** : publié sous **MIT** (interfaces stables : `Stage`, `ContextSpec`, `Result`)  
- **rs3-plugin-fleet** : publié sous **MIT**

## 2. Principe de découplage

Le plugin **Fleet** **n’importe pas directement** de code `core2.*` (AGPL).  
Il se limite à utiliser les interfaces stables fournies par [`rs3-contracts`](https://github.com/SebE585/rs3-contracts) :

```python
from rs3_contracts.api import Stage, ContextSpec, Result
```
L’implémentation concrète (par ex. core2, core3, ou un mock) est résolue à l’exécution
via un adapter chargé dynamiquement (ex. rs3_plugin_fleet.adapters.core2_adapter).

## 3. Conséquence juridique
Sans contrats :
Un import direct from core2.pipeline import PipelineSimulator rendrait Fleet une œuvre dérivée d’un logiciel AGPL → obligation de publier Fleet aussi sous AGPL.
Avec contrats :
Fleet dépend uniquement d’interfaces MIT (rs3-contracts).
L’utilisateur final choisit de l’exécuter avec une implémentation AGPL (core2), mais juridiquement le plugin reste indépendant.
→ Fleet peut donc être publié et distribué sous MIT.

## 4. Responsabilité de l’utilisateur
Si l’utilisateur branche Fleet avec core2 (AGPL), le runtime résultant est globalement couvert par l’AGPL.
Cependant, le code source du plugin Fleet reste MIT, et peut être réutilisé avec toute autre implémentation conforme aux contrats (par ex. un futur core3 sous une licence différente).

## 5. Bonne pratique
Conserver ce découplage : aucun import core2.* dans rs3-plugin-fleet.
Toute dépendance à une implémentation concrète doit passer par un adapter isolé.
Documenter clairement ce choix (ce fichier).

## ✅ Conclusion
rs3-plugin-fleet est juridiquement MIT.
L’exécution avec RS3-core2 se fait via les contrats MIT et un adapter, ce qui évite la « contamination » AGPL.
Ce design garantit la compatibilité multi-implémentations et protège la permissivité de la licence du plugin.