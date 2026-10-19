FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN groupadd --system edition && useradd --system --gid edition --no-create-home edition
COPY --chown=edition:edition backend/ ./backend/
COPY --chown=edition:edition frontend/ ./frontend/
COPY --chown=edition:edition artifacts/model.json artifacts/external_model.json ./artifacts/
COPY --chown=edition:edition research/sources.json ./research/sources.json
USER edition
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=2).read()"]
CMD ["python", "backend/server.py", "--host", "0.0.0.0", "--port", "8080"]
