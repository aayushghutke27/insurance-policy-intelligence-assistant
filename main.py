from dotenv import load_dotenv
from src.document_loader import load_document
from src.text_splitter import split_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import create_retriever
from src.rag_chain import create_rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

documents = load_document(r"C:\Users\Acer\Desktop\AI Engineer\RAG Project\data\policy.pdf")

chunks = split_documents(documents)

embedding = get_embedding_model()

vector_store = create_vector_store(chunks, embedding)

retriever = create_retriever(vector_store)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

rag_chain = create_rag_chain(llm, retriever)

while True:

        question = input("\nAsk about your insurance policy (type exit to quit): ")

        if question.lower() == "exit":
                break

        response = rag_chain.invoke(question)

        print("Assistant:",response)




