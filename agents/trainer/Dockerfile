# The Trainer agent (P4). Build from the repo root:
#   docker compose -f agents/trainer/docker-compose.yml up -d --build
FROM python:3.10-slim
WORKDIR /app
COPY requirements-agents.txt .
RUN pip install --no-cache-dir -r requirements-agents.txt
# only what the agent needs: never COPY . (that would bake .env and its API keys into the image)
COPY shared/ shared/
COPY agents/ agents/
COPY baseline/train.py baseline/train.py
COPY dry_run_jobs/ dry_run_jobs/
COPY harness/ harness/
COPY data/ data/
COPY samples/ samples/
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "agents.trainer", "--db", "/app/runs/board.db"]
