FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    DEFAULT_NUM_BEAMS=1 \
    FORCE_DEVICE=cpu

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY model.py serve_model.py infer.py ./
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
