from langchain_chroma import Chroma


def create_vector_store(chunks, embedding):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db",
        collection_name="insurance_policy"
    )

    return vector_store
