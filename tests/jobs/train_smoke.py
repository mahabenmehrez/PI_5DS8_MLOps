from baseline import train


def run(inputs, workdir):
    df = train.load_wines(inputs["data"])
    res = train.train(df, tracking_uri=f"file:{workdir}/mlruns", reference_dir=workdir / "ref")
    return {"ok": res.version is None, "outputs": {"version": res.version, "run_id": res.run_id}, "metrics": {"mae": res.mae}}
