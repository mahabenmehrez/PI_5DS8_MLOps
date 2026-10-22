# RAG metrics of team mode (from `rag_metrics.py`)

49 diagnoses (every LLM step of the evaluation cases, walked with the verified fixes). Intervals: Wilson 95%, for reference (diagnoses of one fault are not independent).

## Retrieval (advanced retriever, top 3)

- Diagnoses with at least one relevant memory case: 17 of 49
- **Recall@3** (a relevant case in the top 3): 82% (14/17) [59–94%]

## CRAG evaluator: keep or discard each retrieved case

| | relevant | not relevant |
|---|---|---|
| kept | 7 | 0 |
| discarded | 7 | 133 |

- Accuracy: 95% (140/147) [90–98%]
- Precision of the kept cases: 100% (7/7) [65–100%]
- Recall of the relevant retrieved cases: 50% (7/14) [27–73%]

## CRAG action vs the ideal action

Ideal: **correct** when a relevant case was retrieved, else **incorrect** (go to the web). Ambiguous uses both, so it is counted as acceptable either way.

| ideal \ chosen | correct | ambiguous | incorrect |
|---|---|---|---|
| correct | 7 | 0 | 7 |
| incorrect | 0 | 0 | 35 |

- Exact action: 86% (42/49) [73–93%]
- Acceptable action (exact or ambiguous): 86% (42/49) [73–93%]

## Every diagnosis

| Repo | Fault | Error | Relevant in memory | Top 3 (score, kept) | Action |
|---|---|---|---|---|---|
| sk_root | broken_import | ImportError: cannot import name 'LogisticRegresion' from 'sklearn.line | - | gen-006 0.283, gen-002 0.019, gen-003 0.0 | incorrect |
| sk_root | data_dir_path | FileNotFoundError: [Errno 2] No such file or directory: 'iris.csv' | - | gen-006 0.045, gen-002 0.0, gen-003 0.0 | incorrect |
| sk_root | missing_data_file | FileNotFoundError: [Errno 2] No such file or directory: 'data/iris_v2. | - | gen-006 0.043, gen-002 0.0, gen-003 0.0 | incorrect |
| sk_root | numpy_NaN | AttributeError: `np.NaN` was removed in the NumPy 2.0 release. Use `np | gen-003 | gen-003* 0.036, gen-002 0.0, gen-007 0.0 | incorrect |
| sk_root | pandas_df_iteritems | AttributeError: 'DataFrame' object has no attribute 'iteritems'. Did y | - | gen-006 0.019, gen-002 0.0, gen-003 0.0 | incorrect |
| sk_root | pandas_df_iteritems | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_root | pandas_read_csv_squeeze | TypeError: read_csv() got an unexpected keyword argument 'squeeze' | - | gen-003 0.018, gen-006 0.0, fault-001 0.0 | incorrect |
| sk_root | removed_numpy_alias | AttributeError: module 'numpy' has no attribute 'float'. | gen-003 | gen-003* 0.269, gen-002 0.028, gen-006 0.022 | incorrect |
| sk_root | removed_numpy_product | AttributeError: module 'numpy' has no attribute 'product' | gen-003 | gen-003* 0.269, gen-002 0.028, gen-006 0.022 | incorrect |
| sk_root | removed_pandas_append | AttributeError: 'DataFrame' object has no attribute 'append'. Did you  | - | gen-006 0.019, gen-002 0.0, fault-001 0.0 | incorrect |
| sk_root | removed_pandas_append | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_root | removed_scipy_simps | ImportError: cannot import name 'simps' from 'scipy.integrate' (/root/ | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| sk_root | scipy_misc_face | ImportError: cannot import name 'face' from 'scipy.misc' (/root/.cache | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| sk_root | sklearn_bagging_base_estimator | TypeError: BaggingClassifier.__init__() got an unexpected keyword argu | - | gen-003 0.017, fault-001 0.0, gen-006 0.0 | incorrect |
| sk_root | sklearn_load_boston | ImportError: | gen-003 | fault-001 0.0, gen-012 0.0, gen-005 0.0 | incorrect |
| sk_root | sklearn_load_boston | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_root | unpinned_version | TypeError: got an unexpected keyword argument 'squared' | - | fault-001 0.0, gen-003 0.019, gen-007 0.0 | incorrect |
| sk_src | broken_import | ImportError: cannot import name 'RandomForestClassifer' from 'sklearn. | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| sk_src | data_dir_path | FileNotFoundError: [Errno 2] No such file or directory: 'iris.csv' | - | gen-006 0.045, gen-002 0.0, gen-003 0.0 | incorrect |
| sk_src | missing_data_file | FileNotFoundError: [Errno 2] No such file or directory: 'data/iris_v2. | - | gen-006 0.043, gen-002 0.0, gen-003 0.0 | incorrect |
| sk_src | numpy_NaN | AttributeError: `np.NaN` was removed in the NumPy 2.0 release. Use `np | gen-003 | gen-003* 0.036, gen-002 0.0, gen-007 0.0 | incorrect |
| sk_src | pandas_df_iteritems | AttributeError: 'DataFrame' object has no attribute 'iteritems'. Did y | - | gen-006 0.019, gen-002 0.0, gen-003 0.0 | incorrect |
| sk_src | pandas_df_iteritems | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_src | pandas_read_csv_squeeze | TypeError: read_csv() got an unexpected keyword argument 'squeeze' | - | gen-003 0.018, gen-006 0.0, fault-001 0.0 | incorrect |
| sk_src | pandas_read_csv_squeeze | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_src | removed_numpy_alias | AttributeError: module 'numpy' has no attribute 'float'. | gen-003 | gen-003* 0.269, gen-002 0.028, gen-006 0.022 | incorrect |
| sk_src | removed_numpy_product | AttributeError: module 'numpy' has no attribute 'product' | gen-003 | gen-003* 0.269, gen-002 0.028, gen-006 0.022 | incorrect |
| sk_src | removed_pandas_append | AttributeError: 'DataFrame' object has no attribute 'append'. Did you  | - | gen-006 0.019, gen-002 0.0, fault-001 0.0 | incorrect |
| sk_src | removed_pandas_append | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_src | removed_scipy_simps | ImportError: cannot import name 'simps' from 'scipy.integrate' (/root/ | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| sk_src | scipy_misc_face | ImportError: cannot import name 'face' from 'scipy.misc' (/root/.cache | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| sk_src | sklearn_bagging_base_estimator | TypeError: BaggingClassifier.__init__() got an unexpected keyword argu | - | gen-003 0.017, fault-001 0.0, gen-006 0.0 | incorrect |
| sk_src | sklearn_load_boston | ImportError: | gen-003 | fault-001 0.0, gen-012 0.0, gen-005 0.0 | incorrect |
| sk_src | sklearn_load_boston | ValueError: numpy.dtype size changed, may indicate binary incompatibil | gen-003 | gen-003* 1.0 kept, gen-006 0.0, fault-001 0.0 | correct |
| sk_src | unpinned_version | TypeError: got an unexpected keyword argument 'squared' | - | fault-001 0.0, gen-003 0.019, gen-007 0.0 | incorrect |
| wine | broken_import | ImportError: cannot import name 'ElasticNett' from 'sklearn.linear_mod | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| wine | data_dir_path | FileNotFoundError: [Errno 2] No such file or directory: 'winequality-r | - | gen-006 0.043, gen-002 0.0, gen-003 0.0 | incorrect |
| wine | missing_data_file | FileNotFoundError: [Errno 2] No such file or directory: 'data/winequal | - | gen-006 0.042, gen-002 0.0, gen-003 0.0 | incorrect |
| wine | pandas_df_iteritems | AttributeError: 'DataFrame' object has no attribute 'iteritems'. Did y | - | gen-006 0.019, gen-002 0.0, gen-003 0.0 | incorrect |
| wine | pandas_df_iteritems | AttributeError: 'DataFrame' object has no attribute 'iteritems'. Did y | - | gen-006 0.019, gen-002 0.0, gen-003 0.0 | incorrect |
| wine | pandas_read_csv_squeeze | TypeError: read_csv() got an unexpected keyword argument 'squeeze' | - | gen-003 0.018, gen-006 0.0, fault-001 0.0 | incorrect |
| wine | removed_numpy_alias | AttributeError: module 'numpy' has no attribute 'float'. | gen-003 | gen-003* 0.269, gen-002 0.028, gen-006 0.022 | incorrect |
| wine | removed_pandas_append | AttributeError: 'DataFrame' object has no attribute 'append'. Did you  | - | gen-006 0.019, gen-002 0.0, fault-001 0.0 | incorrect |
| wine | removed_pandas_append | AttributeError: 'DataFrame' object has no attribute 'append'. Did you  | - | gen-006 0.019, gen-002 0.0, fault-001 0.0 | incorrect |
| wine | removed_scipy_simps | ImportError: cannot import name 'simps' from 'scipy.integrate' (/root/ | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| wine | scipy_misc_face | ImportError: cannot import name 'face' from 'scipy.misc' (/root/.cache | - | gen-006 0.283, gen-003 0.0, gen-002 0.019 | incorrect |
| wine | sklearn_bagging_base_estimator | TypeError: BaggingClassifier.__init__() got an unexpected keyword argu | - | gen-003 0.017, fault-001 0.0, gen-006 0.0 | incorrect |
| wine | sklearn_load_boston | ImportError: | gen-003 | fault-001 0.0, gen-012 0.0, gen-005 0.0 | incorrect |
| wine | unpinned_version | TypeError: got an unexpected keyword argument 'squared' | - | fault-001 0.0, gen-003 0.019, gen-007 0.0 | incorrect |

`*` = relevant by the rule.
