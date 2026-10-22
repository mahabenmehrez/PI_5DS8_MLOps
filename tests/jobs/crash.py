def run(inputs, workdir):
    print("about to fail")
    raise RuntimeError("boom")
