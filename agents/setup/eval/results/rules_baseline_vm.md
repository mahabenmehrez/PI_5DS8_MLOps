| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 11/15 | 8/12 | 3/3 | 0.53 | 0 | 0 | 3.9 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag |
|---|---|---|---|---|---|
| wine | clean | ✅ 0 |  |  |  |
| wine | runtime_dependency | ✅ 1 (install emoji (the code imports emoji)) |  |  |  |
| wine | unpinned_version | ❌ 0 |  |  |  |
| wine | wrong_data_path | ❌ 0 |  |  |  |
| wine | broken_import | ✅ 0 |  |  |  |
| sk_root | clean | ✅ 0 |  |  |  |
| sk_root | runtime_dependency | ✅ 1 (install emoji (the code imports emoji)) |  |  |  |
| sk_root | unpinned_version | ❌ 0 |  |  |  |
| sk_root | wrong_data_path | ✅ 1 (run train from model/ (iris.csv is there, next to the script) |  |  |  |
| sk_root | broken_import | ✅ 0 |  |  |  |
| sk_src | clean | ✅ 1 (run train as a module (python -m irisclf.train): it uses rel) |  |  |  |
| sk_src | runtime_dependency | ✅ 2 (install emoji (the code imports emoji), run train as a modul) |  |  |  |
| sk_src | unpinned_version | ❌ 0 |  |  |  |
| sk_src | wrong_data_path | ✅ 2 (run train as a module (python -m irisclf.train): it uses rel) |  |  |  |
| sk_src | broken_import | ✅ 0 |  |  |  |
