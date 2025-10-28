# syntax=docker/dockerfile:1
FROM python:3.13 AS builder

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.13
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY . .

EXPOSE 7860
# 7860 on HF spaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]