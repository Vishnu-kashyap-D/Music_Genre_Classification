FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app.py train_parallel_cnn.py train_model_torch.py ./
COPY torch_models/parallel_genre_classifier_torch.pt torch_models/parallel_genre_classifier_torch.pt
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV FLASK_DEBUG=false
EXPOSE 5000

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 120
