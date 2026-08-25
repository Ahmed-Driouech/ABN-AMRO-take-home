# Frontend is built first and served by the API, so the whole application is
# one image, one process and one port.
FROM node:22-slim AS frontend
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache .

COPY backend/ ./backend/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY --from=frontend /app/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
