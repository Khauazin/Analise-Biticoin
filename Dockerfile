# Backend Dockerfile (FastAPI + Uvicorn)
# Uso: docker build -t cryptovision-api .
#      docker run --rm -p 8000:8000 --env ENABLE_BACKGROUND_SERVICE=0 cryptovision-api

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências do sistema (para wheels nativos e SSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Em produção, o loop de background fica desabilitado; rode o worker separadamente se desejar
ENV ENABLE_BACKGROUND_SERVICE=0 \
    BACKGROUND_INTERVAL=30 \
    ALLOW_ORIGINS=*

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

