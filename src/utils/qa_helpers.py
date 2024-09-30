from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def create_retrieval_qa(vector_index):
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=vector_index.as_retriever()
    )

def perform_qa(qa_instance, question):
    return qa_instance.run(question)
