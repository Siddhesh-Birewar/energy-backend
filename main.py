import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import duckdb

app = FastAPI(title="Energy Grid API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "active", "service": "Energy Grid RAG API"}

# Lazy-loaded globals to prevent RAM spike on startup
_rag_chain = None

def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced k to 3 to save memory

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        template = """You are an expert energy research assistant.
Answer the question based ONLY on the following context. If the answer is not contained, say "I do not have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        _rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return _rag_chain

class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
def query_rag(req: QueryRequest):
    try:
        chain = get_rag_chain()
        answer = chain.invoke(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SQLRequest(BaseModel):
    query: str

@app.post("/api/sql")
def execute_sql(req: SQLRequest):
    try:
        con = duckdb.connect("smart_meters.db", read_only=True)
        df = con.execute(req.query).fetchdf()
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))