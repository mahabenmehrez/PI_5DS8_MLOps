"""Send random wine samples to the API so Prometheus and Grafana have traffic to show.

    python baseline/simulate_requests.py              # 1000 requests to http://localhost:8000
    python baseline/simulate_requests.py -n 200 --url http://localhost:8000

Same ranges as simulate_requests.sh, but runs on Windows too (no bash or curl needed).
"""

import argparse
import json
import random
import urllib.request

RANGES = [  # (min, max) per feature, in the API's column order
    (4, 15), (0.1, 1.2), (0, 1), (0.9, 15.5), (0.012, 0.611), (1, 72),
    (6, 289), (0.99007, 1.00369), (2.74, 4.01), (0.33, 2), (8.4, 14.9),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    ok = failed = 0
    for i in range(1, args.n + 1):
        body = json.dumps({"data": [round(random.uniform(lo, hi), 5) for lo, hi in RANGES]}).encode()
        req = urllib.request.Request(f"{args.url}/predict", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                answer = json.load(resp)
            ok += 1
            if i % 100 == 0 or i == 1:
                print(f"request {i}: predicted quality {answer['prediction']:.2f}")
        except Exception as exc:
            failed += 1
            print(f"request {i} failed: {exc}")
    print(f"done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()
