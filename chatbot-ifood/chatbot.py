from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("Carregando dados dos restaurantes...")

loader = TextLoader("restaurantes.txt", encoding="utf-8")
documentos = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documentos)

print("Criando banco vetorial...")
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

print("\n✅ Chatbot pronto! Digite 'sair' para encerrar.\n")
print("Exemplos de perguntas:")
print("  - Qual restaurante japonês você recomenda?")
print("  - Quero algo barato no centro")
print("  - Tem opcao vegana?\n")

while True:
    pergunta = input("Você: ").strip()
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("Ate logo!")
        break
    if not pergunta:
        continue
    resposta = chain.invoke(pergunta)
    print(f"\nChatbot: {resposta}\n")