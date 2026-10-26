FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt && groupadd --system edition && useradd --system --gid edition --create-home edition
COPY --chown=edition:edition backend/ ./backend/
COPY --chown=edition:edition apps/ ./apps/
COPY --chown=edition:edition scripts/run_edition.py ./scripts/run_edition.py
COPY --chown=edition:edition artifacts/model.json artifacts/external_model.json artifacts/commercial_model.json ./artifacts/
COPY --chown=edition:edition config/paywall_policy.json ./config/paywall_policy.json
COPY --chown=edition:edition research/ ./research/
COPY --chown=edition:edition sql/ ./sql/
COPY --chown=edition:edition .streamlit/config.toml ./.streamlit/config.toml
RUN mkdir -p /app/data && chown edition:edition /app/data
USER edition
EXPOSE 8501 4173
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read(); urllib.request.urlopen('http://127.0.0.1:4173/api/health', timeout=2).read()"]
CMD ["python", "scripts/run_edition.py", "--host", "0.0.0.0"]
