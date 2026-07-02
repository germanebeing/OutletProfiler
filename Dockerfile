# Outlet Profiler — single stateful container (FastAPI + in-memory grader +
# in-process worker). Runs as-is on Render / Fly / Railway / Azure Container Apps.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# code
COPY engine ./engine
COPY agent ./agent
COPY api ./api
COPY web ./web
COPY pull_company.py profiler_cli.py image_typing.py clip_classify.py ./
# NB: torch/open_clip are intentionally NOT in the slim serve image, so photo-based
# onboarding degrades cleanly to text typing on the server (works fully on a torch host).

# the base graded dataset + already-onboarded companies (small parquets).
# Grading/validation need no warehouse; only onboarding NEW companies hits Trino.
COPY data/outlets_geo2.parquet ./data/outlets_geo2.parquet
COPY data/onboarded ./data/onboarded

EXPOSE 8100
# Render/Fly inject $PORT; default to 8100 locally.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8100}"]
