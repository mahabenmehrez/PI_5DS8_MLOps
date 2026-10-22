## v1 (15 cases)

| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 12/15 | 9/12 | 3/3 | 0.60 | 0 | 0 | 5.7 |
| llm_plain | 12/15 | 9/12 | 3/3 | 1.20 | 0 | 15 | 9.6 |
| llm_memory | 12/15 | 9/12 | 3/3 | 1.20 | 0 | 15 | 9.0 |
| llm_crag | 15/15 | 12/12 | 3/3 | 1.07 | 0 | 11 | 10.2 |
| llm_sft | 15/15 | 12/12 | 3/3 | 1.07 | 0 | 13 | 10.1 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag | llm_sft |
|---|---|---|---|---|---|---|
| wine | clean | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| wine | runtime_dependency | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) |
| wine | unpinned_version | ❌ 0 | ❌ 3 (pin_package scikit-learn==1.8.0: reverted (it broke the earl) | ❌ 3 (pin_package scikit-learn==1.9.1: reverted (it broke the earl) | ✅ 2 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) | ✅ 2 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) |
| wine | wrong_data_path | ✅ 1 (run train from ./ (data/winequality-red.csv is there, in ./,) | ✅ 1 (run train from ./ (data/winequality-red.csv is there, in ./,) | ✅ 1 (run train from ./ (data/winequality-red.csv is there, in ./,) | ✅ 1 (run train from ./ (data/winequality-red.csv is there, in ./,) | ✅ 1 (run train from ./ (data/winequality-red.csv is there, in ./,) |
| wine | broken_import | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| sk_root | clean | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| sk_root | runtime_dependency | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) |
| sk_root | unpinned_version | ❌ 0 | ❌ 3 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) | ❌ 3 (pin_package scikit-learn==1.9.1: reverted (it broke the earl) | ✅ 2 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) | ✅ 2 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) |
| sk_root | wrong_data_path | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) |
| sk_root | broken_import | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| sk_src | clean | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) |
| sk_src | runtime_dependency | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) |
| sk_src | unpinned_version | ❌ 0 | ❌ 3 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) | ❌ 3 (pin_package scikit-learn==1.8.0: reverted (it broke the earl) | ✅ 3 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) | ✅ 3 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) |
| sk_src | wrong_data_path | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) |
| sk_src | broken_import | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |

## heldout (9 cases)

| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 1/9 | 0/8 | 1/1 | 0.11 | 0 | 0 | 2.5 |
| llm_plain | 1/9 | 0/8 | 1/1 | 1.56 | 0 | 41 | 10.0 |
| llm_memory | 2/9 | 1/8 | 1/1 | 2.22 | 2 | 41 | 13.2 |
| llm_crag | 1/9 | 0/8 | 1/1 | 2.11 | 1 | 44 | 12.5 |
| llm_sft | 1/9 | 0/8 | 1/1 | 2.78 | 3 | 37 | 12.2 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag | llm_sft |
|---|---|---|---|---|---|---|
| wine | removed_numpy_alias | ❌ 0 | ❌ 3 (pin_package numpy==1.25.2, pin_package numpy==1.24.4, pin_pa) | ❌ 2 (pin_package numpy==1.23.5, pin_package numpy==1.24.4: revert) | ❌ 2 (pin_package numpy==1.23.5, pin_package numpy==1.24.4: revert) | ❌ 2 (pin_package numpy==1.23.5, pin_package numpy==1.24.4: revert) |
| wine | removed_pandas_append | ✅ 0 | ✅ 2 (pin_package pandas==2.1.4: reverted (it broke the earlier st) | ✅ 1 (pin_package pandas==2.3.3: reverted (it broke the earlier st) | ✅ 3 (pin_package pandas==2.1.4: reverted (it broke the earlier st) | ✅ 3 (pin_package pandas==2.1.4: reverted (it broke the earlier st) |
| wine | data_dir_path | ❌ 0 | ❌ 0 | ❌ 2 (pin_package pandas==2.1.4: reverted (it broke the earlier st) | ❌ 0 | ❌ 3 (pin_package pandas==2.3.3: reverted (it broke the earlier st) |
| sk_root | removed_numpy_alias | ❌ 0 | ❌ 1 (pin_package numpy==2.3.5: reverted (it broke the earlier sta) | ❌ 3 (pin_package numpy==1.23.5, pin_package numpy==1.24.4: revert) | ❌ 1 (pin_package numpy==2.3.5: reverted (it broke the earlier sta) | ❌ 2 (pin_package numpy==1.23.5, pin_package numpy==1.24.4: revert) |
| sk_root | removed_pandas_append | ❌ 0 | ❌ 3 (pin_package pandas==2.2.3, pin_package pandas==2.3.3, pin_pa) | ❌ 2 (pin_package pandas==2.2.3, pin_package pandas==2.3.3) | ❌ 2 (pin_package pandas==2.2.3, pin_package pandas==2.1.4) | ❌ 3 (pin_package pandas==2.2.3, pin_package pandas==1.5.3, pin_pa) |
| sk_root | data_dir_path | ❌ 0 | ❌ 0 | ✅ 1 (fix_path workdir data) | ❌ 2 (pin_package pandas==2.2.3, pin_package pandas==2.1.4) | ❌ 3 (pin_package pandas==2.2.3, pin_package pandas==2.1.4, pin_pa) |
| sk_src | removed_numpy_alias | ❌ 0 | ❌ 1 (pin_package numpy==2.3.5: reverted (it broke the earlier sta) | ❌ 3 (pin_package numpy==1.23.5, run train as a module (python -m ) | ❌ 3 (pin_package numpy==1.23.5, run train as a module (python -m ) | ❌ 3 (pin_package numpy==2.3.5: reverted (it broke the earlier sta) |
| sk_src | removed_pandas_append | ❌ 0 | ❌ 1 (pin_package pandas==2.2.3) | ❌ 3 (pin_package pandas==2.2.3, pin_package pandas==2.1.4, pin_pa) | ❌ 3 (pin_package pandas==2.2.3, pin_package pandas==2.1.4, pin_pa) | ❌ 3 (pin_package pandas==2.2.3, pin_package pandas==1.5.3, pin_pa) |
| sk_src | data_dir_path | ❌ 1 (run train as a module (python -m irisclf.train): it uses rel) | ❌ 3 (run train as a module (python -m irisclf.train): it uses rel) | ❌ 3 (run train as a module (python -m irisclf.train): it uses rel) | ❌ 3 (run train as a module (python -m irisclf.train): it uses rel) | ❌ 3 (run train as a module (python -m irisclf.train): it uses rel) |

## slots (2 cases)

| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 0/2 | 2/2 | 0/0 | 0.00 | 0 | 0 | 0.6 |
| llm_plain | 2/2 | 2/2 | 0/0 | 0.00 | 0 | 0 | 0.7 |
| llm_memory | 2/2 | 2/2 | 0/0 | 0.00 | 0 | 0 | 0.8 |
| llm_crag | 2/2 | 2/2 | 0/0 | 0.00 | 0 | 0 | 0.8 |
| llm_sft | 2/2 | 2/2 | 0/0 | 0.00 | 0 | 0 | 0.7 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag | llm_sft |
|---|---|---|---|---|---|---|
| sk_two_scripts | clean | ❌ 0 [train_script=train_model.py (rule), target=petal_size (assumed)] | ✅ 0 [train_script=train_model.py (rule), target=species (llm)] | ✅ 0 [train_script=train_model.py (rule), target=species (llm)] | ✅ 0 [train_script=train_model.py (rule), target=species (llm)] | ✅ 0 [train_script=train_model.py (rule), target=species (llm)] |
| sk_legacy | clean | ❌ 0 [train_script=train_old.py (assumed)] | ✅ 0 [train_script=train_v2.py (llm)] | ✅ 0 [train_script=train_v2.py (llm)] | ✅ 0 [train_script=train_v2.py (llm)] | ✅ 0 [train_script=train_v2.py (llm)] |

