# Le Data Checker expliqué

Ce document explique, avec des schémas, ce que fait le projet, le rôle de notre agent, ses
échanges avec les autres agents, le LLM, les techniques de prompt, l'évaluation, la mémoire et le
fine-tuning. Pour les détails techniques d'intégration, voir `INTEGRATION.md`.

---

## 1. Le projet en une page

**Le modèle** prédit la note d'un vin (0 à 10) à partir de 11 mesures chimiques : acidité, sucre,
soufre, pH, alcool… Il tourne derrière une API (FastAPI), il est enregistré dans MLflow et surveillé
avec Prometheus et Grafana.

**Le problème** : un pipeline MLOps classique tourne « à l'aveugle ». Si les données arrivent dans
une mauvaise unité, si le modèle se dégrade, ou si une nouvelle version est moins bonne pour
certains vins, personne ne s'en rend compte tout de suite.

**Notre solution** : **6 agents**, chacun surveille une étape du pipeline, décide avec un LLM
(avec des règles en secours), et ils se parlent par **messages** sur un tableau commun (`board.db`).

```mermaid
flowchart LR
    subgraph Pipeline
        G[Code Git] --> DATA[Lots de données<br/>100 vins] --> TR[Entraînement<br/>MLflow] --> API[API en production] --> PRED[Prédictions<br/>+ vraies notes]
    end
    subgraph Agents
        S[Setup<br/>P1] -.surveille.-> G
        RV[Reviewer<br/>P2] -.surveille.-> G
        C["DATA CHECKER<br/>P3 : nous"] -.surveille.-> DATA
        T[Trainer<br/>P4] -.surveille.-> TR
        D[Deployer<br/>P5] -.surveille.-> API
        M[Monitor<br/>P6] -.surveille.-> PRED
    end
    B[(board.db<br/>messages + tables)]
    S & RV & C & T & D & M <--> B
```

| Agent | Question à laquelle il répond | État |
|---|---|---|
| Setup (P1) | Le dépôt peut-il tourner comme un pipeline ? | en cours |
| Reviewer (P2) | Ce commit change-t-il une variable sans le vouloir ? | en cours |
| **Data Checker (P3)** | **Ce lot de données est-il utilisable ?** | **fini, intégré avec le Trainer** |
| Trainer (P4) | Faut-il réentraîner ? Le nouveau modèle est-il meilleur pour tout le monde ? | fini |
| Deployer (P5) | Quelle part du trafic donner au nouveau modèle ? Faut-il revenir en arrière ? | en cours |
| Monitor (P6) | Le modèle se dégrade-t-il, et à cause de quoi ? | en cours |

---

## 2. Le rôle de notre agent

À chaque nouveau lot de 100 vins, le Data Checker décide **avant** que ces données servent à
l'entraînement :

| Décision | Quand | Exemple |
|---|---|---|
| **Accepter** (`pass`) | rien d'important | lot normal |
| **Accepter et signaler** (`warn`, `pass_with_warning`) | les données sont valides mais différentes | un lot de vins blancs |
| **Réparer** (`repair`) | erreur corrigeable sans risque, testée d'abord sur une copie | 20 % de doublons |
| **Mettre en quarantaine** (`quarantine`) | erreur dangereuse ou irréparable | soufre en g/L au lieu de mg/L |

Il possède aussi des **pièces communes** à toute l'équipe : la boucle de décision des agents
(`shared/base_agent.py`), l'horloge simulée, les tables de données, et le simulateur qui rejoue les
lots et injecte des pannes (`harness/`).

### Comment il pense : la boucle en 7 étapes

```mermaid
flowchart TD
    O[1. Observer<br/>prendre le prochain lot] --> A[2. Analyser<br/>statistiques, sans LLM]
    A -->|rien d'important| P[Accepter directement<br/>pas d'appel au LLM]
    A -->|problème trouvé| DEC[3. Décider<br/>le LLM choisit une action du menu]
    DEC -->|réponse invalide 2 fois| R[Règles de secours]
    DEC --> L{Action risquée ?<br/>niveau 2}
    R --> L
    L -->|oui| DR[4. Dry run<br/>essai sur une copie]
    DR -->|échec| Q[Action la plus sûre :<br/>quarantaine]
    DR -->|réussite| ACT
    L -->|non| ACT[5. Agir<br/>table puis message]
    Q --> ACT
    ACT --> V[6. Vérifier<br/>l'action a-t-elle marché ?]
    V --> MEM[7. Mémoriser<br/>problème, action, résultat]
```

### Ce que l'analyse détecte (étape 2, sans LLM)

| Contrôle | Ce qu'il repère |
|---|---|
| Schéma (Pandera) | valeurs physiquement impossibles (pH de 9) |
| Colonnes | colonne manquante, nouvelle, renommée, ou de type changé |
| Valeurs manquantes | hausse anormale des cases vides |
| Changement d'unité | médiane **et** quartiles multipliés par le même facteur (×10, ×100, ×1000) |
| Doublons | le même vin (`wine_id`) deux fois |
| Dérive | test PSI corrigé du bruit **et** confirmé par le test de Kolmogorov-Smirnov |
| Isolation Forest | lot globalement anormal |

Chaque problème est pondéré par l'**importance de la variable pour le modèle** (venant du Trainer).
Une vraie erreur de données compte toujours, une simple dérive compte selon cette importance.

---

## 3. Les interactions avec les autres agents

Les agents ne s'appellent jamais directement : ils **postent des messages** sur le tableau et
**écrivent l'état dans des tables** (pour qu'un agent redémarré retrouve où il en était).

```mermaid
flowchart LR
    C((Data Checker))
    T[Trainer]
    D[Deployer]
    M[Monitor]
    RV[Reviewer]
    ALL[Tous les agents]
    C -->|data.batch_checked<br/>résultat du contrôle| T
    C -->|request.do_not_use<br/>« n'entraîne pas sur ce lot »| T
    C -->|request.hold<br/>« gèle les mises en production »| D
    C -->|incident.claim<br/>« cet incident est à moi »| ALL
    M -->|query.data_changed<br/>« les données ont changé ? »| C
    T -->|request.clean_window<br/>« quels lots sont propres ? »| C
    RV -->|request.expected_change<br/>« l'unité va changer »| C
```

| Table partagée (`board.db`) | Rôle |
|---|---|
| `batches` | statut de chaque lot (`pending`, `passed`, `repaired`, `quarantined`) et le fichier à entraîner (`path`) |
| `batch_stats` | statistiques de chaque lot (médiane, taux de vides…) |
| `labels` | les vraies notes, qui arrivent avec un jour de retard |
| `claims` | quels incidents notre agent a réclamés (gardé après un redémarrage) |
| `llm_calls` | chaque appel au LLM |

### Un exemple complet : le scénario C

Le fournisseur se met à envoyer le soufre en **g/L au lieu de mg/L** pendant 3 lots.

```mermaid
sequenceDiagram
    participant R as Replay (données)
    participant C as Data Checker
    participant T as Trainer
    participant D as Deployer
    participant M as Monitor
    R->>C: lot 11 (soufre ÷ 1000)
    Note over C: changement d'unité détecté<br/>médiane × 0.0011
    C->>T: request.do_not_use (lot 11)
    C->>M: data.batch_checked : quarantaine
    C->>D: request.hold : gèle les mises en production
    C-->>T: incident.claim : cet incident est à moi
    T->>C: réponse : done, lot exclu
    M->>C: query.data_changed : les données ont changé ?
    C->>M: oui, changement d'unité sur le soufre
    Note over M: donc pas de réentraînement<br/>(il aggraverait le problème)
    R->>C: lots 12 et 13 : même panne
    C->>T: request.do_not_use (même incident, pas de 2e réclamation)
    R->>C: lot 14 : données normales
    C->>T: data.batch_checked : accepté
```

**Résultat** : un seul incident, une seule réclamation, aucun réentraînement inutile, et les lots
redeviennent acceptés dès que la panne s'arrête.

---

## 4. Le LLM utilisé

| | |
|---|---|
| Modèle | **Qwen** `qwen/qwen3.8-27b` (environ 27 milliards de paramètres) |
| Fournisseur | **Groq**, par une API compatible OpenAI |
| Code | `shared/llm.py`, écrit par le Trainer (P4), utilisé par tous les agents |
| Configuration | `agents.yaml` (section `llm`) ; la clé dans `.env`, jamais dans git |
| Réglages | **température 0**, réponse forcée en JSON, contexte limité à **2 000 tokens** |
| Coût mesuré | environ **0.7 s** et **1 100 à 2 100 tokens** par appel |
| Autres fournisseurs possibles | NVIDIA NIM, Ollama (local), ou « off » (règles seules) : un réglage dans la config |

**Ce que le LLM fait, et ne fait pas :**

```mermaid
flowchart LR
    subgraph Code["Le code (déterministe)"]
        A[Mesure tout :<br/>PSI, KS, ratios, taux]
        V[Vérifie la réponse]
        X[Exécute l'action]
    end
    subgraph LLM["Le LLM"]
        CH[Choisit UNE action<br/>du menu + explique]
    end
    A -->|résumé chiffré<br/>jamais les données brutes| CH
    CH -->|JSON| V
    V -->|valide| X
    V -->|invalide 2 fois| RG[Règles de secours] --> X
```

Le LLM ne voit **jamais les données brutes**, seulement des chiffres. C'est important, parce que
l'appel part chez un fournisseur externe.

---

## 5. Les techniques de prompt engineering

Le prompt est dans `agents/checker/prompt.md`. `shared/llm.py` y ajoute le menu et le format de
réponse.

```mermaid
flowchart TD
    subgraph SYS["Message système"]
        R1[Rôle : « Tu es le Data Checker… »]
        R2[Règles : ne jamais mettre en quarantaine pour une simple dérive…]
        R3[Menu : les 7 actions permises et leur niveau]
        R4[Format : JSON avec action, level, params, reason, confidence]
    end
    subgraph USR["Message utilisateur"]
        U1[Contexte chiffré : problèmes trouvés, PSI + rang d'importance,<br/>score d'anomalie, réparations possibles, cas passés similaires]
    end
    SYS --> LLM(("LLM, température 0"))
    USR --> LLM
    LLM --> J[Réponse JSON]
    J --> CHK{Valide ?}
    CHK -->|non| RETRY[Nouvel essai avec l'erreur expliquée] --> LLM
    CHK -->|oui| OK[Décision]
```

| Technique | Ce que ça fait | Pourquoi |
|---|---|---|
| **Role prompting** | le LLM reçoit un rôle précis | il sait quel est son travail |
| **Espace d'action contraint** | il doit choisir dans un menu fixe, au bon niveau | il ne peut pas inventer une action dangereuse |
| **Règles explicites** | « ne jamais réparer ce qui est irréparable »… | on encadre son raisonnement |
| **Sortie structurée (JSON)** | format imposé et mode JSON du fournisseur | le code peut vérifier la réponse automatiquement |
| **Ancrage (grounding)** | il ne reçoit que les chiffres calculés par le code, et doit les citer dans sa raison | il raisonne sur des faits, et on peut contrôler son explication |
| **Mémoire dans le contexte** | les cas passés similaires lui sont fournis | il réutilise ce qui a marché |
| **Auto-correction** | réponse invalide → on la lui renvoie avec l'erreur, une fois | il corrige lui-même son erreur |
| **Repli sur les règles** | 2 réponses invalides → les règles décident | l'agent ne reste jamais bloqué |
| **Température 0** | même question → même réponse | décisions reproductibles |
| **Contexte limité** | au plus 2 000 tokens | coût maîtrisé et pas de données brutes |

C'est du **zero-shot** : le prompt ne contient pas d'exemples de réponses. Ajouter 2 ou 3 exemples
(**few-shot**) serait une amélioration possible.

---

## 6. Les métriques d'évaluation

### Comment on évalue

On **injecte exprès des pannes dont on connaît la bonne réponse**, puis on note les décisions de
l'agent. L'agent ne voit jamais cette « vérité terrain » : elle est dans une table (`faults`) qu'il
ne lit pas.

```mermaid
flowchart LR
    F[8 pannes injectées<br/>× 3 répétitions] --> C[Data Checker]
    N[60 lots normaux] --> C
    C --> D[Décisions]
    GT[(Vérité terrain<br/>table faults)] --> E{Comparaison}
    D --> E
    E --> M1[Taux de bonnes décisions]
    E --> M2[Fausses alarmes]
```

### Les résultats

| Métrique | Ce qu'elle mesure | Résultat | Objectif |
|---|---|---|---|
| **Pannes bien décidées** | la bonne action sur chaque panne | **100 %** (24/24) | ≥ 90 % |
| **Fausses alarmes** | avertissements sur des lots normaux | **0.5 pour 15 lots** | ≤ 1 |
| **Lots normaux bloqués** | lots sains bloqués ou modifiés à tort | **0 sur 60** | 0 |
| **Scénario C** | le bon enchaînement de messages entre agents | **réussi** | réussi |
| **Réponses invalides du LLM** | réponses hors menu ou JSON cassé | **0** | le moins possible |
| **Latence du LLM** | temps de réponse | **~0.7 s** | — |
| **Coût du LLM** | tokens par appel | **~1 100 à 2 100** | — |
| **Tests automatiques** | chaque comportement vérifié | **131 passent** | tous |

Les 8 pannes testées : changement d'unité, valeurs manquantes, colonne renommée, doublons, pH
impossibles, nouvelle colonne, vins blancs, dérive sur une variable peu importante.

**On compare deux systèmes sur les mêmes lots** : les règles seules (le « pipeline traditionnel »)
et le LLM. Sur ces pannes nettes, le LLM prend les mêmes décisions que les règles, avec des
explications justes.

**À mesurer avec les 6 agents réels** : temps de détection, temps de récupération, bon responsable
de l'incident, réentraînements inutiles, actions dangereuses, sollicitations humaines.

---

## 7. La mémoire de l'agent

L'agent utilise **quatre sortes de mémoire**, qui ne durent pas le même temps :

```mermaid
flowchart TD
    subgraph Courte["Le temps d'un tour"]
        M1[Messages reçus<br/>rapport du lot en cours]
    end
    subgraph Experience["Mémoire d'expérience : temporaire"]
        M2[Journal des décisions :<br/>problème, action, a marché ou non]
    end
    subgraph Durable["Mémoire durable : board.db"]
        M3[Tables : statut des lots,<br/>statistiques, incidents réclamés]
        M4[Journal des appels LLM<br/>llm_calls]
    end
    M2 -->|cas similaires| LLM((Contexte du LLM))
    M3 -->|reprise après redémarrage| AG[Agent]
    M4 -->|futur jeu de données| FT[Fine-tuning]
```

| Mémoire | Ce qu'elle garde | Durée | Sert à |
|---|---|---|---|
| **Court terme** | les messages et le rapport du lot en cours | un tour de boucle | décider maintenant |
| **Expérience** (`remember` / `similar`) | chaque problème, l'action choisie, si elle a marché | **tant que l'agent tourne** | donner au LLM les cas passés similaires |
| **État durable** (tables de `board.db`) | statut des lots, statistiques, incidents réclamés | **permanente** | reprendre après un redémarrage, partager avec les autres agents |
| **Journal LLM** (`llm_calls`) | chaque prompt, réponse, tokens, latence | **permanente** | analyser le LLM, futur fine-tuning |

La recherche des cas similaires est **simple** : elle compare les mots des problèmes, sans base
vectorielle. La mémoire d'expérience **durable** (`shared/memory.py`) est confiée au Deployer (P5) :
en attendant, elle est perdue quand l'agent redémarre. Les incidents réclamés, eux, sont déjà gardés
dans la table `claims`.

---

## 8. Le fine-tuning

**On n'a pas fait de fine-tuning.** Le LLM est utilisé tel quel, guidé par le prompt.

**Pourquoi :**
1. **Pas nécessaire pour l'instant** : avec le prompt seul, le LLM prend les bonnes décisions,
   sans réponse invalide.
2. **Pas assez d'exemples** : quelques dizaines d'appels enregistrés, alors qu'il en faut des centaines.
3. **Priorité** : faire marcher les 6 agents ensemble d'abord.

**Ce qui est déjà prêt** : chaque appel est enregistré dans la table `llm_calls` (contexte,
réponse, décision), et on connaît la bonne réponse de chaque panne injectée. C'est exactement le
jeu de données d'un futur fine-tuning.

**La technique qu'on utiliserait :**

```mermaid
flowchart TD
    A[Table llm_calls<br/>contexte + réponse] --> B["Étiqueter avec la bonne<br/>action (vérité terrain)"]
    B --> C[Jeu d'entraînement<br/>contexte → bonne action JSON]
    C --> D[Fine-tuning supervisé SFT<br/>avec LoRA / QLoRA]
    D --> E[Petit modèle ouvert<br/>3 à 8 milliards de paramètres]
    E --> F[Comparer au gros LLM<br/>et aux règles, mêmes pannes]
```

| Élément | Choix | Explication simple |
|---|---|---|
| **Type** | **SFT** (supervised fine-tuning) | on montre au modèle des exemples « situation → bonne décision » |
| **Technique** | **LoRA** ou **QLoRA** | on n'entraîne qu'une petite partie du modèle (des « adaptateurs ») ; QLoRA compresse en plus le modèle pour qu'il tienne sur un ordinateur ordinaire |
| **Modèle** | petit modèle ouvert, par exemple Qwen 3B à 8B | gratuit et utilisable en local avec Ollama |
| **Données** | `llm_calls` + la bonne réponse de chaque panne | aucune donnée de vin brute |
| **Évaluation** | les mêmes 8 pannes et lots normaux | même protocole que ci-dessus |

**L'intérêt** : un petit modèle local, plus rapide, gratuit, sans quota ni envoi de données à
l'extérieur, aussi bon que le gros modèle sur cette tâche précise.

---

## 9. En résumé

```mermaid
mindmap
  root((Data Checker))
    Rôle
      Contrôler chaque lot avant l'entraînement
      Accepter, signaler, réparer, isoler
    Interactions
      Trainer : do_not_use, clean_window
      Deployer : hold
      Monitor : data_changed
      Reviewer : expected_change
    LLM
      Qwen 27B via Groq
      Température 0, JSON
      Règles en secours
    Prompt
      Rôle, menu, règles
      Ancrage sur les chiffres
      Auto-correction
    Évaluation
      100 % des pannes
      0.5 alarme pour 15 lots
      131 tests
    Mémoire
      Expérience, temporaire
      Tables board.db, durable
      Journal LLM
    Fine-tuning
      Pas encore
      SFT + LoRA prévu
```

> **En une phrase :** le code mesure, le LLM décide dans un menu fixe, et le code vérifie avant
> d'agir ; notre agent bloque les vraies pannes sans bloquer les données normales, et il travaille
> déjà avec le Trainer.
