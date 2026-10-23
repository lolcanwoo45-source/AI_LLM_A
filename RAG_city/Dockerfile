FROM python:3.11

작업 디렉토리 설정

WORKDIR /app

pip 최신 버전 업데이트

RUN pip install --no-cache-dir --upgrade pip

★ requirements.txt 파일 없이 도커 안에서 직접 패키지 설치 ★

파일 누락으로 인한 빌드 실패를 원천 차단합니다.

RUN pip install --no-cache-dir 

fastapi 

uvicorn 

python-multipart 

python-dotenv 

langchain 

langchain-core 

langchain-community 

langchain-openai 

langchain-google-genai 

supabase 

pymupdf 

tiktoken

나머지 전체 코드(app.py 등) 복사

COPY . .

포트 설정

ENV PORT=8080
EXPOSE 8080

서버 실행

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
