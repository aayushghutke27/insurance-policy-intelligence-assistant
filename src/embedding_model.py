from langchain_google_genai import GoogleGenerativeAIEmbeddings



def get_embedding_model():
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    return embedding
