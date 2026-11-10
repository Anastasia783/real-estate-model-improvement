FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "SQLAlchemy==2.0.36" \
    && pip check \
    && python -c "from sqlalchemy.pool import FallbackAsyncAdaptedQueuePool"

EXPOSE 5000
