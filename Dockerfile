FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8050

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY requirements-app.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-app.txt

COPY app ./app
COPY config ./config
COPY models ./models
COPY data/sample ./data/sample
RUN mkdir -p data/monitoring && chown -R appuser:appgroup /app

USER appuser
EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health', timeout=3)" || exit 1

CMD ["gunicorn", "--chdir", "app", "--bind", "0.0.0.0:8050", "--workers", "1", "--threads", "4", "--timeout", "120", "app:server"]
