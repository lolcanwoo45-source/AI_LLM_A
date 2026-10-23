FROM python:3.10-slim

작업 디렉토리 설정

WORKDIR /app

PyMuPDF 및 기타 라이브러리 빌드/실행에 필요한 시스템 의존성 패키지 설치

RUN apt-get update && apt-get install -y 

build-essential 

libgl1-mesa-glx 

libglib2.0-0 

&& rm -rf /var/lib/apt/lists/*

pip 최신 버전 업데이트 (휠(wheel) 파일 설치 오류 방지)

RUN pip install --no-cache-dir --upgrade pip

요구사항 파일 복사 및 설치

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

소스 코드 복사

COPY . .

구글 클라우드 런 포트 설정

ENV PORT=8080
EXPOSE 8080

FastAPI 서버 실행

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
