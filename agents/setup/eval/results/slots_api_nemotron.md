## slots (2 cases)

| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| llm_crag | 2/2 | 2/2 | 0/0 | 0.00 | 0 | 2 | 1.1 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag | llm_sft |
|---|---|---|---|---|---|---|
| sk_two_scripts | clean |  |  |  | ✅ 0 [train_script=train_model.py (rule), target=species (llm)] |  |
| sk_legacy | clean |  |  |  | ✅ 0 [train_script=train_v2.py (llm)] |  |

