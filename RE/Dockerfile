Python 3.10 슬림 버전 사용

FROM python:3.10-slim

작업 디렉토리 설정

WORKDIR /app

시스템 패키지 업데이트 및 빌드 필수 도구 설치 (ChromaDB 컴파일 등에 필요할 수 있음)

RUN apt-get update && apt-get install -y 

build-essential 

&& rm -rf /var/lib/apt/lists/*

요구사항 파일 복사 및 패키지 설치

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

애플리케이션 소스 코드 복사

COPY . .

포트 설정 (구글 클라우드 런 기본 포트는 8080)

EXPOSE 8080

Uvicorn을 사용해 FastAPI 서버 실행

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
