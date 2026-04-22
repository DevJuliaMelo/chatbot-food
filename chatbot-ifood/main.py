from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    loader = TextLoader("restaurantes.txt", encoding="utf-8")
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)

    embeddings = FakeEmbeddings(size=1536)
    banco_vetorial = FAISS.from_documents(chunks, embeddings)
    retriever = banco_vetorial.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    prompt = PromptTemplate.from_template("""Você é um assistente de recomendação de restaurantes do iFood.
Use apenas as informações abaixo para responder. Seja simpático e objetivo.
Se não souber a resposta com base nos dados, diga que não encontrou.

Contexto:
{context}

Pergunta: {question}
Resposta:""")

    def formatar_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | formatar_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Pergunta(BaseModel):
    pergunta: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/perguntar")
def perguntar(body: Pergunta):
    resposta = chain.invoke(body.pergunta)
    return {"resposta": resposta}
