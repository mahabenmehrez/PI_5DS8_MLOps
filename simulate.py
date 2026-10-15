import random, time, requests

url = "http://localhost:8000/predict"
lat = []
errors = 0
for _ in range(1000):
    data = [round(random.uniform(4, 12), 2) for _ in range(11)]
    t = time.time()
    r = requests.post(url, json={"data": data})
    lat.append(time.time() - t)
    if r.status_code != 200:
        errors += 1
lat.sort()
print("requêtes:", len(lat), "erreurs:", errors)
print("moyenne (ms):", round(1000 * sum(lat) / len(lat), 1))
print("p95 (ms):", round(1000 * lat[int(0.95 * len(lat))], 1))