## final (17 cases)

Invalid (the fault doesn't fail the build), left out: wine/numpy_NaN

| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 2/17 | 0/15 | 2/2 | 0.00 | 0 | 0 | 3.3 |
| llm_crag | 8/17 | 6/15 | 2/2 | 1.18 | 0 | 47 | 13.9 |
| llm_sft | 10/17 | 8/15 | 2/2 | 1.53 | 0 | 50 | 15.3 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag | llm_sft |
|---|---|---|---|---|---|---|
| wine | scipy_misc_face | ❌ 0 |  |  | ✅ 1 (pin_package scipy==1.14.1) | ✅ 1 (pin_package scipy==1.14.1) |
| wine | pandas_read_csv_squeeze | ✅ 0 |  |  | ✅ 0 | ✅ 0 |
| wine | sklearn_bagging_base_estimator | ❌ 0 |  |  | ❌ 0 | ❌ 0 |
| wine | pandas_df_iteritems | ✅ 0 |  |  | ✅ 1 (pin_package pandas==2.3.3: reverted (it broke the earlier st) | ✅ 1 (pin_package pandas==2.3.3: reverted (it broke the earlier st) |
| wine | sklearn_load_boston | ❌ 0 |  |  | ❌ 0 | ✅ 3 (pin_package scikit-learn==1.2.2, pin_package scikit-learn==0) |
| sk_root | scipy_misc_face | ❌ 0 |  |  | ✅ 1 (pin_package scipy==1.14.1) | ✅ 1 (pin_package scipy==1.14.1) |
| sk_root | pandas_read_csv_squeeze | ❌ 0 |  |  | ❌ 0 | ❌ 0 |
| sk_root | sklearn_bagging_base_estimator | ❌ 0 |  |  | ❌ 0 | ❌ 0 |
| sk_root | numpy_NaN | ❌ 0 |  |  | ✅ 1 (pin_package numpy==1.26.4) | ✅ 1 (pin_package numpy==1.26.4) |
| sk_root | pandas_df_iteritems | ❌ 0 |  |  | ❌ 1 (pin_package pandas==2.2.3) | ❌ 2 (pin_package pandas==2.2.3, pin_package pandas==2.1.4) |
| sk_root | sklearn_load_boston | ❌ 0 |  |  | ❌ 0 | ✅ 3 (pin_package scikit-learn==1.2.2, pin_package numpy==1.23.5, ) |
| sk_src | scipy_misc_face | ❌ 0 |  |  | ✅ 2 (pin_package scipy==1.14.1, run train as a module (python -m ) | ✅ 2 (pin_package scipy==1.14.1, run train as a module (python -m ) |
| sk_src | pandas_read_csv_squeeze | ❌ 0 |  |  | ❌ 2 (pin_package pandas==2.2.3, pin_package pandas==2.1.4) | ❌ 3 (pin_package pandas==2.1.4, pin_package pandas==2.0.3, pin_pa) |
| sk_src | sklearn_bagging_base_estimator | ❌ 0 |  |  | ✅ 3 (change_entry_command train module, pin_package scikit-learn=) | ✅ 3 (pin_package scikit-learn==1.2.2, pin_package numpy==1.23.5, ) |
| sk_src | numpy_NaN | ❌ 0 |  |  | ✅ 2 (pin_package numpy==1.22.4, run train as a module (python -m ) | ✅ 2 (pin_package numpy==1.26.4, run train as a module (python -m ) |
| sk_src | pandas_df_iteritems | ❌ 0 |  |  | ❌ 3 (change_entry_command train module, change_entry_command trai) | ❌ 1 (pin_package pandas==2.1.4) |
| sk_src | sklearn_load_boston | ❌ 0 |  |  | ❌ 3 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) | ❌ 3 (pin_package scikit-learn==1.6.1, pin_package scikit-learn==1) |

