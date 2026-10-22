import os


def run(inputs, workdir):
    print("echo job running")
    return {"ok": inputs.get("ok", True), "outputs": {"dry_run": os.environ.get("DRY_RUN"), "cwd": str(workdir)},
            "metrics": {"x": inputs.get("x", 0) * 2}}
