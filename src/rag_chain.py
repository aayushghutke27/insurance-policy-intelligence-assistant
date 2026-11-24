
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

prompt = ChatPromptTemplate.from_template(
    """You are an insurance policy assistant.
           
        Answer the user's question using ONLY the provided
        policy context.

        If the answer is not available in the context, 
        clearly say that the policy document does not provide this information

        User Question : {question}

        Policy Context: {context}


        Answer:

        """
)


parser = StrOutputParser()


def format_documents(documents):

    format_documents = []

    for doc in documents:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page_number = page + 1
        else:
            page_number = page

        formatted_text = f"""
            Source:{source} | Page:{page_number}
            {doc.page_content}
        """

        format_documents.append(formatted_text)

    return "\n\n".join(format_documents)


def create_rag_chain(llm, retriever):

    rag_chain = ({

        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(format_documents)

    }
        | prompt | llm | parser)

    return rag_chain
