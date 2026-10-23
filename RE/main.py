import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain 및 Google GenAI 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정 (클라이언트 측 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 보안을 위해 실제 서비스 시에는 GAS 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 구글 API 키 환경변수 확인
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY가 설정되지 않았습니다. API 호출 시 오류가 발생할 수 있습니다.")

# 전역 벡터 스토어 (데모용, 실제 운영 시 외부 저장소 또는 Persistent Volume 사용 권장)
CHROMA_PERSIST_DIR = "./chroma_db"
vectorstore = None

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "서버가 정상적으로 실행 중입니다."}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트를 추출, 청크로 분할 후 Chroma DB에 저장합니다."""
    global vectorstore
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"PDF 로드 시작: {file.filename}")
        
        # 1. PDF 텍스트 변환 및 로드
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 2. 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"총 {len(chunks)}개의 청크로 분할되었습니다.")

        # 3. 구글 임베딩 모델 사용 (models/text-embedding-004)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # 4. 벡터 DB (Chroma)에 적용
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # 임시 파일 삭제
        os.remove(tmp_path)

        return {"message": f"{file.filename} 파일이 성공적으로 RAG DB에 추가되었습니다.", "chunks_count": len(chunks)}
    
    except Exception as e:
        logger.error(f"PDF 업로드 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_stream(request: QueryRequest):
    """사용자의 질의를 받아 MultiQueryRetriever로 검색 후, LLM 스트리밍 답변을 반환합니다."""
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="먼저 PDF 파일을 업로드하여 DB를 생성하세요.")

    try:
        # 언어 모델 설정 (실시간 스트리밍 지원)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, streaming=True)

        # 5. 다중 질문 검색기 (MultiQueryRetriever) 적용
        # 사용자의 질문을 다양한 각도로 변형하여 검색 품질을 높입니다.
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            llm=llm
        )

        # 6. 프롬프트 허브 사용 및 시스템 역할 설정
        # 프롬프트 허브에서 RAG 템플릿을 가져온 후, 현진건 작가의 페르소나를 시스템 메시지로 덮어씌웁니다.
        base_prompt = hub.pull("rlm/rag-prompt")
        
        persona_system_message = (
            "당신은 소설 운수 좋은날을 집필한 현진건 작가님입니다. "
            "근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요. "
            "주어진 문맥(Context)을 바탕으로 대답하되, 특유의 구슬프고 사실적인 말투를 사용하시오.\n\n"
            "Context: {context}"
        )
        
        # 새로운 프롬프트 템플릿 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", persona_system_message),
            ("human", "{question}")
        ])

        # 문서 포맷팅 함수
        def format_docs(docs):
            return "\n\n".join(f"[출처: {doc.metadata.get('page', '알수없음')}페이지] {doc.page_content}" for doc in docs)

        # 7. 질의 확장 (RunnablePassthrough) 및 체인 구성 (LCEL)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        # 8. 실시간 스트리밍 제너레이터 함수
        async def generate():
            # 문맥(Context)을 먼저 클라이언트에게 전송하여 '출처 표기' 영역에 표시할 수 있도록 커스텀 포맷 적용
            docs = retriever.get_relevant_documents(request.question)
            context_str = format_docs(docs)
            
            # 클라이언트 단에서 컨텍스트와 LLM 답변을 분리하기 위한 구분자 전송
            yield f"__CONTEXT__\n{context_str}\n__ENDCONTEXT__\n"

            # LLM 스트리밍 (StreamHandler 역할)
            async for chunk in rag_chain.astream(request.question):
                yield chunk.content

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"채팅 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
