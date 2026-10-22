## cascade (16 cases)

Invalid (the fault doesn't fail the build), left out: wine/numpy_asfarray, wine/numpy_unicode_

| Config | Success | Fixable built | Unfixable reported | Mean attempts | Unrelated installs | LLM calls | Minutes |
|---|---|---|---|---|---|---|---|
| rules | 1/16 | 0/15 | 1/1 | 0.00 | 0 | 0 | 3.1 |
| llm_sft | 7/16 | 6/15 | 1/1 | 1.25 | 0 | 38 | 13.4 |
| llm_cascade | 10/16 | 9/15 | 1/1 | 1.31 | 0 | 45 | 18.0 |

| Repo | Fault | rules | llm_plain | llm_memory | llm_crag | llm_sft | llm_cascade |
|---|---|---|---|---|---|---|---|
| wine | scipy_trapz | ❌ 0 |  |  |  | ❌ 2 (pin_package scipy==1.14.1, pin_package scipy==1.15.3) | ✅ 2 (pin_package scipy==1.14.1, pin_package scipy==1.13.1) |
| wine | pandas_error_bad_lines | ✅ 0 |  |  |  | ✅ 0 | ✅ 0 |
| wine | scipy_binom_test | ❌ 0 |  |  |  | ❌ 0 | ❌ 0 |
| wine | numpy_int_alias | ❌ 0 |  |  |  | ✅ 3 (pin_package numpy==1.25.2, pin_package numpy==1.24.4, pin_pa) | ❌ 1 (pin_package numpy==1.25.2) |
| sk_root | scipy_trapz | ❌ 0 |  |  |  | ✅ 3 (pin_package scipy==1.14.1, pin_package scipy==1.15.3, pin_pa) | ✅ 2 (pin_package scipy==1.14.1, pin_package scipy==1.13.1) |
| sk_root | pandas_error_bad_lines | ❌ 0 |  |  |  | ❌ 0 | ❌ 0 |
| sk_root | scipy_binom_test | ❌ 0 |  |  |  | ❌ 0 | ❌ 0 |
| sk_root | numpy_unicode_ | ❌ 0 |  |  |  | ✅ 1 (pin_package numpy==1.25.2) | ✅ 1 (pin_package numpy==1.25.2) |
| sk_root | numpy_int_alias | ❌ 0 |  |  |  | ❌ 0 | ✅ 1 (pin_package numpy==1.23.5) |
| sk_root | numpy_asfarray | ❌ 0 |  |  |  | ✅ 1 (pin_package numpy==1.25.2) | ✅ 1 (pin_package numpy==1.25.2) |
| sk_src | scipy_trapz | ❌ 0 |  |  |  | ❌ 3 (pin_package scipy==1.14.1, pin_package scipy==1.15.3, pin_pa) | ✅ 3 (pin_package scipy==1.14.1, pin_package scipy==1.11.4, run tr) |
| sk_src | pandas_error_bad_lines | ❌ 0 |  |  |  | ❌ 3 (pin_package pandas==2.1.4, pin_package pandas==2.0.3, pin_pa) | ❌ 3 (pin_package pandas==2.1.4, pin_package pandas==1.4.4, pin_pa) |
| sk_src | scipy_binom_test | ❌ 0 |  |  |  | ❌ 0 | ❌ 0 |
| sk_src | numpy_unicode_ | ❌ 0 |  |  |  | ✅ 2 (pin_package numpy==1.25.2, run train as a module (python -m ) | ✅ 2 (pin_package numpy==1.25.2, run train as a module (python -m ) |
| sk_src | numpy_int_alias | ❌ 0 |  |  |  | ❌ 0 | ✅ 3 (pin_package numpy==1.26.4, pin_package numpy==1.23.5, run tr) |
| sk_src | numpy_asfarray | ❌ 0 |  |  |  | ✅ 2 (pin_package numpy==1.25.2, run train as a module (python -m ) | ✅ 2 (pin_package numpy==1.25.2, run train as a module (python -m ) |

