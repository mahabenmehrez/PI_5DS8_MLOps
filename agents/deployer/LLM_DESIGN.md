# Le LLM du Deployer (P5) : conception, prompt et évaluation

## 1. Quel est le rôle du LLM ?

Le Deployer décide quelle part du trafic reçoit un nouveau modèle (shadow, 10 %, 50 %, puis 100 % avec un humain).
La plupart des décisions sont **déterministes** (du code) : c'est un schéma connu (canary release), et le code est
plus rapide, moins cher et testable.

Le LLM n'intervient que dans la **zone grise** :

| Situation | Qui décide |
|---|---|
| Limite dure dépassée (erreurs > 5 % ou p95 > 300 ms pendant 2 min) | **Code** : rollback automatique, sans LLM ni approbation |
| Toutes les portes passées | **Code** : avance, ou demande l'humain à 50 % |
| Données insuffisantes | **Code** : attend |
| Une porte franchement négative, mais aucune limite dure | **LLM** : `hold` ou `rollback` |
| Passage à 100 % | **Humain**, toujours |

## 2. Stratégie de prompt

Le prompt système est dans `agents/deployer/prompt.md`. Il suit six choix :

1. **Rôle et périmètre clairs** : qui il est, quand il est appelé, ce qu'il ne décide jamais.
2. **Menu d'actions fermé** : seulement `hold` ou `rollback`, avec leur niveau.
3. **Stratégie de décision explicite** : gravité, convergence des signaux, étape (shadow ou trafic réel), hold demandé
   par un autre agent, mémoire des cas passés, coût de l'erreur. Par défaut en cas de doute : `hold`.
4. **Contexte en chiffres uniquement** : étape, latence p95, taux d'erreurs, portes négatives, holds, cas passés,
   limites dures. Jamais de lignes brutes de données ; le message est plafonné à environ 2 000 jetons.
5. **Sortie JSON imposée** : `action`, `level`, `params`, `reason`, `confidence`, `success_check`. Le `reason` doit
   citer les chiffres décisifs ; le `success_check` est écrit avant d'agir, pour pouvoir vérifier la décision.
6. **Deux exemples** (un `hold`, un `rollback`) pour fixer le format et le niveau d'exigence (few-shot).

Réglages : température 0, sortie JSON forcée (Ollama `format: json`, API en JSON dans le prompt).

## 3. Garde-fous

| Risque | Parade |
|---|---|
| Réponse non JSON ou incomplète | validation stricte, **un** nouvel essai avec le message d'erreur, puis repli sur les règles |
| Action hors menu, ou interdite (avancer, promouvoir, rollback d'urgence) | rejetée à la validation **et** par un deuxième garde-fou dans l'agent |
| Mauvais niveau pour l'action | rejeté |
| Modèle indisponible, clé API absente | exception capturée, repli sur les règles |
| Appels répétés inutiles | une décision identique est réutilisée pendant 10 minutes simulées |
| Limite dure | le LLM n'est **jamais** appelé : le code agit |
| Promotion à 100 % | niveau 3 : l'agent demande à un humain, quoi que dise le LLM |

Conséquence : **le LLM ne peut jamais rendre le système moins sûr** que les règles. Il peut seulement choisir entre
attendre et retirer le candidat.

## 4. Traçabilité

Chaque appel est journalisé : prompt envoyé, réponse brute, décision validée, validité, erreur, jetons, latence
(`DeployerLLM.calls`, et un fichier JSONL si `DEPLOYER_LLM_LOG` est défini). Chaque action de l'agent est aussi
mémorisée dans l'experience log, et les cas passés alimentent les prompts suivants. Ce journal pourrait servir plus
tard de jeu de données pour affiner un modèle.

## 5. Esprit de l'évaluation

Question posée : **le LLM apporte-t-il quelque chose par rapport aux règles seules, sans jamais être dangereux ?**

- **Jeu de cas** : 8 situations de zone grise écrites à la main (`evaluation/eval_deployer_llm.py`), chacune avec les
  actions jugées acceptables (lenteur légère, latence 10 fois trop élevée, erreurs 4 %, peu de données, shadow,
  historique de rollbacks, hold demandé par un autre agent, erreurs à 50 % du trafic).
- **Mesures** :
  - taux de réponses valides ;
  - taux de décisions acceptables (LLM et règles seules, pour comparer) ;
  - tentatives dangereuses (objectif : bloquées à 100 %) ;
  - latence et jetons par décision (coût).
- **Répétitions** : 3 par cas, parce que les réponses d'un LLM varient.
- **Comparaison** : règles seules (toujours `hold`) contre LLM local (Ollama) contre LLM par API.
- **Tests automatiques** (`tests/test_deployer_llm.py`) : validation, nouvel essai, actions interdites, repli,
  absence d'appel hors zone grise, absence d'appel sur limite dure, réutilisation de la décision.

### Limites à reconnaître
- Huit cas écrits à la main ne couvrent pas la réalité ; les « actions acceptables » sont un jugement humain.
- Dans la démo, la zone grise apparaît surtout avec le scénario « Candidat moins précis ».
- Le LLM n'a pas été affiné ; seuls le prompt et le contexte sont réglés.
- Un modèle local de 3 milliards de paramètres sera moins constant qu'un modèle par API : c'est justement ce que
  l'évaluation mesure.

## 6. Pour tester

```bash
python -m pytest tests                                        # tests, sans LLM réel
python -m evaluation.eval_deployer_llm --provider fake         # vérifie le script d'évaluation
python -m evaluation.eval_deployer_llm --provider ollama --repeats 3
python -m evaluation.eval_deployer_llm --provider api --repeats 3
streamlit run demo_deployer_live.py                            # choisir "Décision dans la zone grise"
```
