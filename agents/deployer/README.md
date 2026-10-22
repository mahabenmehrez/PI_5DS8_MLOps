\# Baseline : mesures et étapes manuelles (P5, Deployer)



\## Mémoire (docker stats, stack Wine seule)

| Conteneur | Repos | Sous charge |

|---|---|---|

| mlflow\_server | 632 MiB | 633 MiB |

| grafana | 558 MiB | 558 MiB |

| mlflow\_db | 405 MiB | 406 MiB |

| app (API) | 209 MiB | 210 MiB |

| mlflow\_s3 (MinIO) | 108 MiB | 110 MiB |

| prometheus | 69 MiB | 70 MiB |

| Total | \~1,98 GiB | \~1,98 GiB |



Pas de fuite mémoire visible. CPU de l'API : \~22 % sous charge.



\## Latence de /predict (1000 requêtes séquentielles, simulate.py)

\- Série 1 : moyenne 17,7 ms, p95 24,7 ms, 0 erreur

\- Série 2 : moyenne 13,2 ms, p95 21,6 ms, 0 erreur

Le seuil de rollback prévu (p95 > 300 ms) est très au-dessus.



\## Bug de predict() reproduit

Requête : {"data": \[1,2]} (2 valeurs au lieu de 11)

Résultat : erreur 500, UnboundLocalError: local variable 'prediction\_value'

referenced before assignment (main.py, ligne 154).

Cause : le return est dans le bloc finally ; si le modèle échoue,

prediction\_value n'existe pas et l'erreur d'origine est masquée.

Laissé tel quel volontairement (scénario H).



\## Étapes manuelles nécessaires pour lancer le baseline

1\. minio/minio et minio/mc ne sont plus sur Docker Hub (Quay renvoie 401) :

&#x20;  image remplacée par cgr.dev/chainguard/minio.

2\. Service mc remplacé par un conteneur amazon/aws-cli qui crée le bucket "mlflow".

3\. Healthcheck du service db décommenté (le service web attend service\_healthy).

4\. wait-for-it.sh converti de CRLF en LF (sinon "bash\\r: No such file").

5\. Python 3.13 incompatible avec mlflow 2.14.1 : entraînement lancé dans

&#x20;  un conteneur python:3.10-slim, sur le réseau example-network.

6\. Modèle entraîné et enregistré "wine-quality" ; alias champion posé sur la

&#x20;  version 1 (la version doit être passée en texte).

7\. app/main.py : chargement par alias models:/wine-quality@champion.

8\. L'API attend {"data": \[11 nombres]}, pas une liste seule.

9\. Warning : mlflow 2.9.2 dans l'API contre 2.14.1 pour le modèle.

10\. Conteneurs de l'ancien projet (Airflow) arrêtés pour ne pas fausser les mesures.



\## Livrables partagés

\- shared/memory.py : experience log SQLite (remember, similar, best\_fixes), testé.

\- samples/predictions.parquet : 400 prédictions simulées (stand-in pour le Monitor).

