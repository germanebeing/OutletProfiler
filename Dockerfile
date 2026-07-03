# Outlet Profiler — single stateful container (FastAPI + in-memory grader +
# in-process worker). Runs as-is on Render / Fly / Railway / Azure Container Apps.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
# vision deps for photo-based outlet typing (SigLIP). CPU-only torch wheel keeps
# the image off the multi-GB CUDA build. open_clip is pinned (the model API +
# labels are tied to it); Pillow decodes the downloaded storefront photos.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir open_clip_torch==3.3.0 pillow==12.3.0

# code
COPY engine ./engine
COPY agent ./agent
COPY api ./api
COPY web ./web
COPY pull_company.py profiler_cli.py image_typing.py clip_classify.py ./
# SigLIP weights (~400MB) download from the open_clip hub on the first photo job.

# the base graded dataset + already-onboarded companies (small parquets).
# Grading/validation need no warehouse; only onboarding NEW companies hits Trino.
COPY data/outlets_geo2.parquet ./data/outlets_geo2.parquet
COPY data/onboarded ./data/onboarded

EXPOSE 8100
# Render/Fly inject $PORT; default to 8100 locally.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8100}"]
