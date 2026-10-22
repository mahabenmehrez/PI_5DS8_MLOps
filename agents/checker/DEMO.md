# Démo du Data Checker (P3) — script de présentation

Durée : **10 à 12 minutes**. Chaque étape donne la commande, ce que tu vas voir et ce que tu dis.
Les textes à dire sont dans les encadrés « 🗣️ ».

---

## Avant la séance (5 min, à faire chez toi)

1. Ouvre un terminal **Git Bash** dans le dossier du projet :

```bash
cd "C:/Users/jalle/Desktop/Esprit/5DS8/PI/New folder/Pi-Mlops"
```

2. Vérifie que tout marche (environ 2 minutes, les tests du Trainer entraînent des modèles) :

```bash
.venv/Scripts/python -m pytest -q
```

Tu dois voir `131 passed` (tes tests, ceux du Trainer et ceux de l'intégration). Les « warnings » viennent de la bibliothèque mlflow : ce ne sont pas des erreurs.

3. Pour que les accents s'affichent bien dans la console, tape une seule fois :

```bash
export PYTHONIOENCODING=utf-8
```

4. Vérifie que ton fichier `.env` (la clé Groq) est bien là, pour l'étape 5. Ne l'affiche **jamais** à l'écran.

5. Garde ouverts dans l'éditeur : `agents/checker/INTEGRATION.md` (le schéma) et `agents/checker/prompt.md`.

---

## Le plus simple : l'interface web

Au lieu de taper les commandes, lance une seule fois :

```bash
.venv/Scripts/python -m harness.demo_ui
```

Puis ouvre **http://127.0.0.1:8765** dans ton navigateur.

La page contient les étapes 2 à 4 ci-dessous (numérotées 1 à 3 sur la page), avec **un bouton par
étape**. Les tests automatiques (étape 1) n'y sont pas : si on te pose une question sur la fiabilité,
réponds que 131 tests vérifient chaque comportement après chaque modification. Le texte à dire n'est
pas affiché sur la page (elle est faite pour être projetée) : garde ce fichier ouvert à côté.
En haut, un sélecteur **« Les règles / Le LLM »** choisit qui prend la décision (le LLM n'est
disponible que si ton `.env` contient la clé).

Chaque clic fait tourner le vrai code de l'agent : rien n'est préenregistré. Pour arrêter : `Ctrl+C`
dans le terminal. La page ne répond qu'à ton ordinateur (adresse `127.0.0.1`).

La suite de ce fichier reste valable si tu préfères les commandes, ou en secours.

---

## Étape 0 — Présenter le problème (1 min, sans commande)

> 🗣️ « Notre projet est un pipeline MLOps qui prédit la qualité d'un vin à partir de 11 mesures
> chimiques. Six agents surveillent chacun une partie du pipeline et communiquent par messages.
> Moi, je suis responsable du **Data Checker**.
>
> Son rôle : à chaque fois qu'un lot de 100 vins arrive, décider si on peut l'utiliser. Le
> problème, c'est que de mauvaises données passent sans bruit : une unité qui change, des valeurs
> manquantes, des doublons. Le modèle continue de prédire, mais il prédit faux, et on ne le voit
> pas tout de suite.
>
> Mon agent a 4 réponses possibles : **accepter**, **réparer**, **mettre en quarantaine**, ou
> **accepter en signalant un changement**. »

Si tu veux, montre le schéma en haut de `INTEGRATION.md` (§1) : le lot arrive, le Checker décide, et il prévient le Trainer, le Deployer et le Monitor.

---

## Étape 1 — Les tests automatiques (1 min)

```bash
.venv/Scripts/python -m pytest -q
```

**Tu verras :** `131 passed`.

> 🗣️ « D'abord, la qualité du code : 131 tests automatiques, les miens, ceux du Trainer et ceux qui font tourner nos deux agents ensemble. Ils vérifient la boucle de décision
> commune aux 6 agents, que j'ai écrite, l'horloge partagée, chaque contrôle et chaque réparation,
> et ce qui se passe quand le LLM répond n'importe quoi. Dans ce cas, l'agent revient
> automatiquement à des règles. Il ne s'arrête jamais. »

---

## Étape 2 — La démo principale : trois lots, trois décisions (3 min)

```bash
.venv/Scripts/python -m harness.demo_checker
```

**Tu verras trois blocs.** Pour chacun, l'agent reçoit 10 lots normaux puis un lot en panne.

### Cas 1 — doublons → **RÉPARÉ**

Lignes à montrer : `duplicates {'rate': 0.1667}`, `Décision : repair`, `Essai sur une copie (dry run) : réussi`.

> 🗣️ « Premier lot : 20 % de lignes en double. C'est réparable sans risque. L'agent supprime les
> doublons, mais **d'abord sur une copie**. C'est le dry run. Il vérifie que la copie réparée
> passe tous les contrôles, et seulement là, il garde le lot réparé. Le fichier d'origine n'est
> jamais modifié. »

### Cas 2 — soufre en g/L au lieu de mg/L → **MIS EN QUARANTAINE**

Lignes à montrer : `unit_change … median_ratio 0.0011, factor 0.001`, et les 4 messages envoyés.

> 🗣️ « Deuxième lot : le fournisseur envoie le soufre en grammes au lieu de milligrammes, donc
> toutes les valeurs sont 1000 fois trop petites. L'agent le détecte parce que la médiane **et**
> les quartiles ont été divisés par 1000 en même temps. Une vraie évolution ne ferait jamais ça.
>
> Ici, il ne répare **pas**. Cette erreur touche aussi l'API en production, qui reçoit les mêmes
> mauvaises valeurs. Réparer en cachette masquerait le problème. Alors il met le lot en
> quarantaine, il dit au Trainer de ne pas l'utiliser, il prend la responsabilité de l'incident
> pour que les autres agents ne fassent pas le même travail, et il demande au Deployer de
> bloquer les mises en production. »

### Cas 3 — vins blancs → **ACCEPTÉ avec avertissement**

Lignes à montrer : `drift volatile acidity {'psi': 4.102, 'ks_p': 0.0}`, `Décision : pass_with_warning`.

> 🗣️ « Troisième lot : des vins blancs. Les chiffres sont très différents du vin rouge, mais ils
> sont **physiquement valides**. Ce n'est pas une panne, c'est le monde qui change. L'agent
> l'accepte et prévient les autres agents. C'est au Trainer de décider s'il faut réentraîner le
> modèle. Bloquer ce lot serait une erreur. »

**La phrase clé à dire après les 3 cas :**

> 🗣️ « Donc mon agent ne répare pas tout. Il répare ce qui est sûr, il bloque ce qui est
> dangereux, et il laisse passer ce qui est simplement nouveau. »

---

## Étape 3 — Un scénario complet avec plusieurs agents (2 min)

```bash
.venv/Scripts/python -m harness.scenario_runner C
```

**Tu verras :** la liste des messages échangés, heure par heure, puis `scenario C: PASS` à la fin.

Lignes à montrer, dans l'ordre :
1. `11:00 checker -> ... data.batch_checked  Batch batch-0011 quarantined: unit_change…`
2. `11:00 checker -> ... incident.claim  Data incident: unit_change on total_sulfur_dioxide`
3. `12:00 monitor -> checker  query.data_changed  Live MAE up since batch 11…`
4. `12:00 checker -> monitor  query.data_changed  Yes: unit_change on total sulfur dioxide`

> 🗣️ « Ici, c'est le scénario C de notre roadmap, avec les autres agents. Pour l'instant, ils
> sont simulés, en attendant l'intégration. La panne dure 3 lots. À 11 h, mon agent bloque le
> lot et réclame l'incident. À midi, le Monitor voit que les erreurs du modèle augmentent et
> demande : est-ce que les données ont changé ? Mon agent répond : oui, changement d'unité.
>
> Grâce à cette réponse, le Monitor ne demande **pas** de réentraîner le modèle. Réentraîner sur
> des données fausses aurait aggravé le problème. Un seul incident, une seule réclamation, et
> quand la panne s'arrête au lot 14, les lots sont de nouveau acceptés. »

---

## Étape 4 — Les résultats chiffrés (2 min)

Cette commande prend environ 30 secondes. Lance-la, puis parle pendant qu'elle tourne :

```bash
.venv/Scripts/python -m harness.evaluate_checker --seeds 3 --normal 60
```

**Tu verras :** une ligne `ok` par panne, puis :

```
faults decided correctly: 100%   (target >= 90%)
normal batches: 60, wrongly blocked/repaired: 0, passed with a warning: 2  -> 0.5 alarms per 15 (target <= 1)
```

> 🗣️ « Pour mesurer, on injecte 8 types de pannes, chacune répétée 3 fois avec des lots
> différents. L'agent prend la bonne décision dans 100 % des cas. Notre objectif était 90 %.
>
> Il faut aussi vérifier l'inverse : qu'il ne bloque pas les bonnes données. Sur 60 lots
> normaux, il n'en a bloqué ou modifié **aucun**. Il a juste signalé 3 petites dérives, soit
> 0.5 alarme pour 15 lots, pour un objectif de 1 au maximum. »

**Ce qu'on a appris en testant** (très bon point à dire) :

> 🗣️ « Au début, on avait plus de fausses alarmes. Avec seulement 100 vins par lot, les
> statistiques bougent beaucoup d'un lot à l'autre : la médiane du soufre varie de 20 % sans
> aucune panne. On a donc adapté les seuils à la taille du lot, et on confirme chaque dérive par
> un test statistique, le test de Kolmogorov-Smirnov. Autre découverte : le modèle de base
> n'utilise presque que le degré d'alcool. Si on jugeait la gravité seulement par l'importance de
> la variable, une erreur sur le soufre serait ignorée. On a donc séparé les vraies erreurs de
> données, qui comptent toujours, des simples dérives, qui comptent selon l'importance. »

---

## Étape 5 — Le LLM prend la décision (2 min)

```bash
.venv/Scripts/python -m harness.demo_checker --llm
```

**Tu verras :** `Décisions par le LLM : groq / qwen/qwen3.8-27b`, puis les mêmes 3 cas avec `(décidé par le LLM)` et ses explications en anglais.

> 🗣️ « Jusqu'ici, c'étaient des règles. Maintenant, c'est un vrai LLM, Qwen, via l'API Groq, qui
> choisit l'action. Les statistiques sont toujours calculées par du code. Le LLM ne voit **que
> des chiffres**, jamais les données brutes, et il doit choisir une action dans un menu fixe, en
> JSON.
>
> Regardez ses explications : il cite les bons chiffres, par exemple le ratio 0.0011 pour le
> soufre. Et s'il répond n'importe quoi, l'agent refuse la réponse, lui redonne une chance une
> fois, puis revient aux règles. Sur nos essais : 0 réponse invalide. »

Si on te demande pourquoi garder les règles :

> 🗣️ « Les règles servent de sécurité, et de base de comparaison : la roadmap nous demande de
> comparer un pipeline classique, avec des règles, à notre système d'agents. »

⚠️ Si internet ou Groq ne répond pas pendant la séance, saute cette étape. Dis que l'agent revient automatiquement aux règles, ce qui est justement la sécurité prévue.

---

## Étape 6 — Conclusion et suite (1 min)

> 🗣️ « Pour résumer : mon agent contrôle chaque lot avant l'entraînement. Il répare ce qui est
> sûr, il bloque ce qui est dangereux, et il laisse passer les vraies évolutions. Il atteint les
> objectifs fixés, et un LLM peut prendre la décision avec des garde-fous.
>
> J'ai aussi écrit des pièces communes à toute l'équipe : la boucle de décision des agents,
> l'horloge simulée, les tables de données, et le simulateur qui rejoue les lots et injecte les
> pannes.
>
> Prochaine étape : l'intégration avec les vrais agents de mes coéquipiers la semaine prochaine,
> en commençant par le Monitor, puis la comparaison règles contre LLM en semaine 4. »

---

## Questions probables et réponses

**« Pourquoi ne pas tout réparer automatiquement ? »**
> Une réparation douteuse transforme une erreur visible en erreur cachée. Pour un changement
> d'unité, rien ne prouve que *toutes* les valeurs sont dans la mauvaise unité. Bloquer est
> plus sûr, et l'incident est signalé.

**« Comment l'agent sait-il que c'est un changement d'unité et pas une vraie évolution ? »**
> Une unité qui change multiplie *toutes* les valeurs par le même facteur (10, 100 ou 1000) : la
> médiane et les deux quartiles bougent du même facteur. Une vraie évolution les déplace de façon
> inégale. Et le facteur doit être une puissance de 10 : une variation naturelle n'atteint jamais ×10.

**« Que se passe-t-il si le LLM se trompe ? »**
> Trois protections. Il ne peut choisir qu'une action du menu, au bon niveau, sinon sa réponse est
> refusée. Une réparation est toujours testée d'abord sur une copie. Et les actions graves
> demandent une validation humaine (niveau 3).

**« Le LLM reçoit-il les données ? »**
> Non. Seulement des chiffres : taux, ratios, PSI, p-values. Jamais de lignes de données. C'est
> important parce que l'appel part chez un fournisseur externe.

**« Les autres agents sont-ils vraiment là ? »**
> Pas encore : ils sont simulés par des « faux agents » qui envoient des messages scriptés. Mais
> j'utilise exactement les mêmes formats de messages et de fonctions que ceux définis dans la
> roadmap, donc l'intégration consistera à remplacer les faux par les vrais, sans changer mon agent.

**« Pourquoi des lots de 100 vins ? »**
> Le jeu de données complet fait 1 599 vins rouges. On le rejoue par lots de 100 pour simuler
> l'arrivée de nouvelles données au fil du temps. C'est la taille prévue par la roadmap.

**« Quelles sont les limites ? »**
> Les seuils ont été réglés sur ce jeu de données. Le taux de fausses alarmes mesuré est un peu
> optimiste, parce que la référence et les lots viennent du même fichier en attendant les données
> du Trainer. Et l'intégration réelle avec les autres agents reste à faire.

---

## Si quelque chose ne marche pas pendant la démo

| Problème | Solution |
|---|---|
| `No module named ...` | Tu n'es pas dans le dossier `Pi-Mlops`, ou tu as lancé `python` au lieu de `.venv/Scripts/python` |
| Caractères bizarres dans la console | Retape `export PYTHONIOENCODING=utf-8` |
| Étape 5 : erreur `GROQ_API_KEY is not set` | Le fichier `.env` manque : saute l'étape 5 |
| Étape 5 : très lent (15 à 25 s) | C'est le quota gratuit de Groq qui fait attendre : patiente, ça repart tout seul |
| Étape 4 : tu veux aller plus vite | Garde quand même `--seeds 3 --normal 60` (30 s). Sur moins de lots, le chiffre d'alarmes n'est pas fiable : un seul lot fait tout bouger, et tu pourrais afficher un résultat au-dessus de l'objectif |
