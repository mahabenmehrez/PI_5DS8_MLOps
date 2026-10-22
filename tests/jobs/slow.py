import time


def run(inputs, workdir):
    time.sleep(inputs.get("sleep", 5))
    return {"ok": True}
