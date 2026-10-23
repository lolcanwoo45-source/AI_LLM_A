FROM python:3.10-slim

2. 작업 디렉토리 설정

WORKDIR /app

3. 시스템 의존성 패키지 설치 (PyMuPDF 등 C 라이브러리 빌드용)

RUN apt-get update && apt-get install -y 

build-essential 

&& rm -rf /var/lib/apt/lists/*

4. 요구사항 파일 복사 및 설치

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

5. 소스 코드 복사

COPY . .

6. 구글 클라우드 런 포트 설정

ENV PORT=8080
EXPOSE 8080

7. FastAPI 서버 실행

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
