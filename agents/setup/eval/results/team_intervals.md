# Team-mode evaluation: 95% intervals (cluster bootstrap over faults)

Development sets (v1, heldout after v2, heldout_v3 after v3) were used to change the agent; only `final` is an unseen test. Few faults per set: wide intervals are the honest picture.

## team_eval_v1

### v1 (15 cases, 5 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 11/15 (73%) | 33–100% |
| llm_plain | 9/15 (60%) | 27–93% |
| llm_memory | 9/15 (60%) | 27–93% |
| llm_crag | 8/15 (53%) | 13–93% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_plain − rules | -13% | -40% to +0% | no |
| llm_memory − rules | -13% | -40% to +0% | no |
| llm_crag − rules | -20% | -60% to +0% | no |

## team_eval_v2

### v1 (15 cases, 5 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 12/15 (80%) | 40–100% |
| llm_plain | 12/15 (80%) | 40–100% |
| llm_memory | 12/15 (80%) | 40–100% |
| llm_crag | 15/15 (100%) | 100–100% |
| llm_sft | 15/15 (100%) | 100–100% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_plain − rules | +0% | +0% to +0% | no |
| llm_memory − rules | +0% | +0% to +0% | no |
| llm_crag − rules | +20% | +0% to +60% | no |
| llm_sft − rules | +20% | +0% to +60% | no |

Fine-tuned − base (llm_sft − llm_crag): +0%, 95% +0% to +0% (includes 0); better on 0 cases, worse on 0

### heldout (9 cases, 3 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 1/9 (11%) | 0–33% |
| llm_plain | 1/9 (11%) | 0–33% |
| llm_memory | 2/9 (22%) | 0–33% |
| llm_crag | 1/9 (11%) | 0–33% |
| llm_sft | 1/9 (11%) | 0–33% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_plain − rules | +0% | +0% to +0% | no |
| llm_memory − rules | +11% | +0% to +33% | no |
| llm_crag − rules | +0% | +0% to +0% | no |
| llm_sft − rules | +0% | +0% to +0% | no |

Fine-tuned − base (llm_sft − llm_crag): +0%, 95% +0% to +0% (includes 0); better on 0 cases, worse on 0

### slots (2 cases)

Too few faults for an interval: rules 0/2, llm_plain 2/2, llm_memory 2/2, llm_crag 2/2, llm_sft 2/2

## team_eval_v3

### v1 (15 cases, 5 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 12/15 (80%) | 40–100% |
| llm_crag | 15/15 (100%) | 100–100% |
| llm_sft | 15/15 (100%) | 100–100% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_crag − rules | +20% | +0% to +60% | no |
| llm_sft − rules | +20% | +0% to +60% | no |

Fine-tuned − base (llm_sft − llm_crag): +0%, 95% +0% to +0% (includes 0); better on 0 cases, worse on 0

### heldout (9 cases, 3 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 1/9 (11%) | 0–33% |
| llm_crag | 6/9 (67%) | 33–100% |
| llm_sft | 8/9 (89%) | 67–100% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_crag − rules | +56% | +0% to +100% | no |
| llm_sft − rules | +78% | +33% to +100% | yes |

Fine-tuned − base (llm_sft − llm_crag): +22%, 95% +0% to +33% (includes 0); better on 2 cases, worse on 0

### slots (2 cases)

Too few faults for an interval: rules 0/2, llm_crag 2/2, llm_sft 2/2

### heldout_v3 (9 cases, 3 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 4/9 (44%) | 0–100% |
| llm_crag | 4/9 (44%) | 0–100% |
| llm_sft | 4/9 (44%) | 0–100% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_crag − rules | +0% | +0% to +0% | no |
| llm_sft − rules | +0% | +0% to +0% | no |

Fine-tuned − base (llm_sft − llm_crag): +0%, 95% +0% to +0% (includes 0); better on 0 cases, worse on 0

## team_eval_final

### final (17 cases, 6 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 2/17 (12%) | 0–24% |
| llm_crag | 8/17 (47%) | 22–76% |
| llm_sft | 10/17 (59%) | 39–82% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_crag − rules | +35% | +6% to +71% | yes |
| llm_sft − rules | +47% | +17% to +81% | yes |

Fine-tuned − base (llm_sft − llm_crag): +12%, 95% +0% to +35% (includes 0); better on 2 cases, worse on 0

## team_eval_cascade

### cascade (16 cases, 6 faults)

| Config | Success | 95% interval |
|---|---|---|
| rules | 1/16 (6%) | 0–19% |
| llm_sft | 7/16 (44%) | 22–71% |
| llm_cascade | 10/16 (62%) | 31–93% |

| Difference vs rules | Point | 95% interval | Excludes 0 |
|---|---|---|---|
| llm_sft − rules | +38% | +11% to +71% | yes |
| llm_cascade − rules | +56% | +24% to +93% | yes |

Cascade − fine-tuned alone (llm_cascade − llm_sft): +19%, 95% +0% to +41% (includes 0); better on 4 cases, worse on 1

