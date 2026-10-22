| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 11/15 | 8/12 | 3/3 | 0.53 | 0 | 0 | 4.9 |
| llm_plain | 9/15 | 8/12 | 1/3 | 1.27 | 4 | 35 | 10.9 |
| llm_memory | 9/15 | 8/12 | 1/3 | 1.13 | 4 | 30 | 10.8 |
| llm_crag | 8/15 | 8/12 | 0/3 | 1.20 | 4 | 34 | 11.1 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag |
|---|---|---|---|---|---|
| wine | clean | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| wine | runtime_dependency | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) |
| wine | unpinned_version | ❌ 0 | ❌ 1 (pin_package sklearn==2.9.2: reverted (it broke the earlier s) | ❌ 3 (pin_package mlflow==2.9.2, pin_package numpy==1.23.5, pin_pa) | ❌ 1 (pin_package scikit-learn==2.9.2: reverted (it broke the earl) |
| wine | wrong_data_path | ❌ 0 | ❌ 2 (fix_path workdir baseline, pin_package pandas==2.2.3) | ❌ 1 (fix_path workdir baseline) | ❌ 2 (fix_path workdir data, pin_package pandas==2.2.3) |
| wine | broken_import | ✅ 0 | ❌ 2 (pin_package scikit-learn==1.7.2, fix_path workdir baseline) | ❌ 1 (pin_package scikit-learn==1.7.2) | ❌ 1 (pin_package scikit-learn==1.7.2) |
| sk_root | clean | ✅ 0 | ✅ 0 | ✅ 0 | ✅ 0 |
| sk_root | runtime_dependency | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) | ✅ 1 (install emoji (the code imports emoji)) |
| sk_root | unpinned_version | ❌ 0 | ❌ 1 (pin_package scikit-learn==1.7.2) | ❌ 2 (pin_package scikit-learn==1.7.2, pin_package scikit-learn==1) | ❌ 3 (pin_package scikit-learn==1.7.2, pin_package scikit-learn==1) |
| sk_root | wrong_data_path | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) | ✅ 1 (run train from model/ (iris.csv is there, next to the script) |
| sk_root | broken_import | ✅ 0 | ❌ 1 (pin_package scikit-learn==1.7.2) | ✅ 0 | ❌ 1 (pin_package scikit-learn==1.7.2) |
| sk_src | clean | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) |
| sk_src | runtime_dependency | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) | ✅ 2 (install emoji (the code imports emoji), run train as a modul) |
| sk_src | unpinned_version | ❌ 0 | ❌ 1 (pin_package scikit-learn==1.7.2) | ❌ 1 (pin_package scikit-learn==1.7.2) | ❌ 1 (pin_package scikit-learn==1.6.2: reverted (it broke the earl) |
| sk_src | wrong_data_path | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) |
| sk_src | broken_import | ✅ 0 | ✅ 3 (change_entry_command train module, change_entry_command trai) | ❌ 1 (pin_package scikit-learn==1.7.2) | ❌ 1 (pin_package scikit-learn==1.7.2) |
