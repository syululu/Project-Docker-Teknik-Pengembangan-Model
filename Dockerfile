
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

FROM base AS api

COPY api.py .
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui

COPY ui.py .
COPY model/ ./model/
COPY heart_disease_uci.csv .

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.port=8501"]