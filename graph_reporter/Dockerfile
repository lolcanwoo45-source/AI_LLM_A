# ==============================================================================
# Google Cloud Run 배포용 Dockerfile
# PulseDesk AI (graph_reporter) - LangGraph 신문기자 심층 분석 기사 생성 서버
# ==============================================================================
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

