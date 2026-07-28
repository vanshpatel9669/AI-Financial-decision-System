FROM python:3.14-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# The flagship model is committed to the repo (models/) so the API is
# runnable out of the box; it is not retrained at build time. To serve a
# different algorithm/reward-mode combination, retrain with
# scripts/train.py and set FINRL_ALGORITHM / FINRL_REWARD_MODE.
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "finrl.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
