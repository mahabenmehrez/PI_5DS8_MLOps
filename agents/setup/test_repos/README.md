# Practice repos for the Setup Agent (roadmap 7.1)

Tiny scikit-learn projects with different layouts, written to test the agent on more than the wine project:

| Repo | Layout | What the agent has to work out |
|---|---|---|
| `sk_root/` | `train.py` and `predict.py` at the root, a `requirements.txt` | nothing special: should build first time |
| `sk_src/` | a package in `src/irisclf/`, **no requirements file** | the dependencies from the imports; that `train.py` uses a relative import, so it must run as a module (`python -m irisclf.train`), not as a file |
| `sk_two_scripts/` | `train_model.py` (the model, saved) and `train_baseline.py` (a side experiment on another column, saves nothing); the data has an extra column `petal_size` | the decide step: the rule picks the script that saves a model; the target (`species` or `petal_size`, same evidence) needs the LLM, which sees the training script's lines. Truth: `train_model.py`, `species` |
| `sk_legacy/` | `train_v2.py` (current) and `train_old.py` (kept for comparison), both save a model | the training script needs the LLM (the rule's guess, the first by name, is `train_old.py`). Truth: `train_v2.py` |

The two decide-step repos (added for team_eval v2) have their ground truth in `eval/team_eval.py` (`SLOT_TRUTH`):
a build with the wrong choice counts as a failure.

Data: the iris dataset (150 rows, from scikit-learn) as `data/iris.csv`; target column `species`.
