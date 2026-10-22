"""Subprocess entry point for shared.dry_run. Not called directly.

    python -m shared._dry_run_entry <job.py> <workdir>
"""

import json
import sys
from pathlib import Path

from shared.dry_run import load_job


def main() -> int:
    job, workdir = Path(sys.argv[1]), Path(sys.argv[2])
    inputs = json.loads((workdir / "inputs.json").read_text())
    result = load_job(job).run(inputs, workdir)
    (workdir / "result.json").write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
