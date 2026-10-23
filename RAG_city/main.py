import os
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# LangChain 컴포넌트 임포트
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 로드
load_dotenv()

# 환경 변수 유효성 검사 (키 누락 시 명확한 에러 발생)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("필수 환경 변수(SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)가 설정되지 않았습니다.")

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 임베딩 모델 (OpenAI) 및 벡터 스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# LLM 모델 초기화 (Gemini 3.5 Flash Lite)
# 스트리밍을 지원하기 위해 구성됩니다. LangChain에서는 비동기 제너레이터(astream)를 사용하여 스트리밍합니다.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2 # 사실 기반 답변을 위해 온도를 낮춤
)

app = FastAPI(title="Google Cloud Run RAG Server")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    입력한 PDF 파일을 텍스트로 변환하고 청크 단위로 쪼개어 Supabase 벡터 DB에 저장하는 엔드포인트
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        # 1. 임시 파일로 PDF 저장 (PyMuPDFLoader는 파일 경로가 필요함)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # 2. PDF 로드 및 텍스트 추출
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()

        # 3. 텍스트 분할 (청크 단위 분할)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # 4. 벡터 DB에 저장 (OpenAI 임베딩 자동 수행 후 Supabase로 전송)
        vector_store.add_documents(split_docs)

        # 임시 파일 삭제
        os.remove(tmp_file_path)

        return {"message": "PDF 처리가 완료되어 벡터 데이터베이스에 성공적으로 저장되었습니다.", "chunks_count": len(split_docs)}

    except Exception as e:
        # 에러 메시지 세밀화
        raise HTTPException(status_code=500, detail=f"PDF 처리 및 벡터 DB 저장 중 서버 오류가 발생했습니다: {str(e)}")


@app.post("/chat")
async def chat_with_docs(request_data: Dict[str, Any]):
    """
    사용자의 질의와 대화 내역을 받아 문서를 검색하고 스트리밍으로 답변을 생성하는 엔드포인트
    요청 바디 예시:
    {
        "query": "그 정책의 주요 효과는 뭐야?",
        "chat_history": [
            {"role": "user", "content": "서울시 청년 수당 정책에 대해 알려줘"},
            {"role": "ai", "content": "서울시 청년 수당은..."}
        ]
    }
    """
    user_query = request_data.get("query")
    history_data = request_data.get("chat_history", [])

    if not user_query:
        raise HTTPException(status_code=400, detail="사용자 질의(query)가 제공되지 않았습니다.")

    try:
        # 1. 대화 기록(Chat History) 포맷팅
        formatted_history = []
        for msg in history_data:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "ai":
                formatted_history.append(AIMessage(content=msg.get("content")))

        # 2. 시스템 메시지 및 프롬프트 설계 (질의 확장용)
        # MessagesPlaceholder를 사용하여 과거 대화 내역이 통째로 들어갈 공간 마련
        query_augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
                       "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 3. Query Augmentation Chain 구성
        # 프롬프트 -> LLM -> 텍스트 파서(StrOutputParser) 파이프라인
        query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()

        # 4. 체인 실행 (데이터 주입)
        if formatted_history:
            augmented_query = await query_augmentation_chain.ainvoke({
                "chat_history": formatted_history,
                "question": user_query
            })
        else:
            augmented_query = user_query # 대화 내역이 없으면 원본 질문 사용

        # 5. 증강된 질의로 벡터 DB 검색 (Retriever)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = await retriever.ainvoke(augmented_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 6. 최종 답변 생성을 위한 시스템 메시지 설정 및 프롬프트
        final_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n"
                       "참고 문서:\n{context}"),
            ("human", "{question}")
        ])

        # 최종 응답 체인 구성
        rag_chain = final_qa_prompt | llm | StrOutputParser()

        # 7. 실시간 스트리밍 핸들러 구성 (비동기 제너레이터)
        async def generate_stream():
            try:
                # 스트리밍 방식(astream)으로 청크를 하나씩 클라이언트에 전달
                async for chunk in rag_chain.astream({
                    "context": context,
                    "question": augmented_query
                }):
                    yield chunk
            except Exception as stream_err:
                yield f"\n[스트리밍 중 에러 발생: {str(stream_err)}]"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        # 에러 메시지 세밀화
        raise HTTPException(status_code=500, detail=f"AI 응답 생성 및 스트리밍 처리 중 서버 오류가 발생했습니다: {str(e)}")
