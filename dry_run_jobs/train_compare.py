"""Dry-run job train_compare (owner: Trainer, roadmap 5.10).

inputs:  db_path, train_ids, eval_ids, now, tracking_uri, live_model_uri,
         min_gain_pct, max_group_worse_pct, resamples
outputs: run_id of the candidate (logged in the dry_run experiment, not registered),
         the comparison report and the gate verdict
"""

import json

from agents.trainer.analysis import compare, gate
from agents.trainer.data import TrainerStore
from baseline import train


def run(inputs, workdir):
    store = TrainerStore(inputs["db_path"])
    train_df = store.rows(inputs["train_ids"], inputs.get("now"))
    eval_df = store.rows(inputs["eval_ids"], inputs.get("now"))
    if len(train_df) < inputs.get("min_rows", 200) or len(eval_df) == 0:
        return {"ok": False, "outputs": {"error": "not enough labelled rows"},
                "metrics": {"train_rows": len(train_df), "eval_rows": len(eval_df)}}

    result = train.train_window(
        train_df, eval_df,
        live_model_uri=inputs.get("live_model_uri", "models:/wine-quality@champion"),
        tracking_uri=inputs.get("tracking_uri"),
        tags={"train_batches": ",".join(inputs["train_ids"]), "eval_batches": ",".join(inputs["eval_ids"])},
    )
    frame = result["frame"].rename(columns={"prediction": "pred_new"})
    report = compare(frame, resamples=inputs.get("resamples", 500))
    verdict = gate(report, min_gain_pct=inputs.get("min_gain_pct", 2.0),
                   max_group_worse_pct=inputs.get("max_group_worse_pct", 5.0))

    import mlflow
    mlflow.MlflowClient().log_text(result["run_id"], json.dumps({"report": report, "gate": verdict}, indent=2),
                                   "comparison.json")
    print(f"candidate run {result['run_id']}: gain {verdict['overall_gain_pct']}%, "
          f"failing groups {[(g['group'], g['band']) for g in verdict['failing_groups']]}")
    return {
        "ok": True,
        "outputs": {"run_id": result["run_id"], "report": report, "gate": verdict,
                    "comparison_ref": f"runs:/{result['run_id']}/comparison.json"},
        "metrics": {"train_rows": len(train_df), "eval_rows": len(eval_df),
                    "mae_new": report["overall"]["mae_new"], "mae_live": report["overall"]["mae_live"]},
    }
