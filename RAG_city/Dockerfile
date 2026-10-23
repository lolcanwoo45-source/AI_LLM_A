FROM python:3.11-slim

WORKDIR /app

pip 최신 버전 업데이트

RUN pip install --no-cache-dir --upgrade pip

requirements.txt 복사 및 설치

시스템 패키지 설치 없이 파이썬 패키지(바이너리 휠)만 설치하도록 단순화

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

나머지 소스코드 복사

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
