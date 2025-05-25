#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def main():
    print("Hello, world!")

    # Load environment variables from .env filepip 
    load_dotenv()

    # Set Open API key
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    try:
        # 1. Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Embeddings created successfully.")

        # 2. Load vector store from disk
        vector_store = load_vector_store(embeddings);
        if vector_store is None:
            # Create vector store if it does not exist
            print("Vector store not found. Creating a new vector store...")
            vector_store = create_vector_store(embeddings)

        # 3. Retrival QA
        retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriver,
        )

        # 4. Ask a question
        question = "踏み台のHDD情報を教えてください?"
        answer = qa.invoke(question)
        print(f"Answer: {answer}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def load_vector_store(embeddings):
    """Load the vector store from disk."""
    try:
        loaded_vector_store = FAISS.load_local("vector/faiss_index", embeddings)
        print("Vector store loaded successfully.")
        return loaded_vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None
    
def create_vector_store(embeddings):
    """Create a vector store from the documents."""

    # 1. Load PDF
    loader = PyPDFLoader("docs/sample.pdf")
    docs = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(f"Number of chunks: {len(texts)}")
    print(f"First chunk: {texts[0]}")

    # 4. Create vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully.")

    # 5. Save vector store to disk
    vector_store.save_local("vector/faiss_index")
    print("Vector store saved to disk.")

    return vector_store


if __name__ == "__main__":
    main()