FROM python:3.11-slim

작업 디렉토리 설정

WORKDIR /app

시스템 필수 패키지 설치 (PyMuPDF, tiktoken 등 설치 시 C++ 컴파일러 에러 방지)

RUN apt-get update && apt-get install -y --no-install-recommends 

gcc 

g++ 

build-essential 

&& rm -rf /var/lib/apt/lists/*

패키지 파일 복사

COPY requirements.txt .

pip 업그레이드 및 파이썬 패키지 설치

RUN pip install --no-cache-dir --upgrade pip && 

pip install --no-cache-dir -r requirements.txt

나머지 애플리케이션 코드 복사

COPY . .

환경 변수 및 포트 설정

ENV PORT=8080
EXPOSE 8080

실행 명령어

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
